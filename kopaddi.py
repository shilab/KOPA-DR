import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from transformers import LlamaForCausalLM
from process_kge import load_pretrain_kge

# ====== KoPATrainer and helpers (add to kopa.py) ======
import torch
import torch.nn.functional as F
from transformers import Trainer

def _resolve_ent_rel_embeddings(model: nn.Module):
    """
    从 KoPAWithAdapter 实例拿到实体/关系嵌入权重。
    优先调用 KoPAWithAdapter.get_entity_relation_weights()。
    """
    if hasattr(model, "get_entity_relation_weights"):
        return model.get_entity_relation_weights()
    # 兜底：尝试直接从 .embeddings 里解析
    embs = getattr(model, "embeddings", None)
    if embs is None:
        raise RuntimeError("Model has no .embeddings")
    ent_w = getattr(getattr(embs, 'ent_embeddings', None), 'weight', None)
    rel_w = getattr(getattr(embs, 'rel_embeddings', None), 'weight', None)
    if ent_w is None and isinstance(embs, dict):
        ent_w = embs.get('ent_embeddings.weight', None)
    if rel_w is None and isinstance(embs, dict):
        rel_w = embs.get('rel_embeddings.weight', None)
    if ent_w is None or rel_w is None:
        raise RuntimeError("Cannot resolve entity/relation weights")
    return ent_w, rel_w

def _distmult_score(h, r, t):
    # h,r,t: [B, D] -> [B]
    return (h * r * t).sum(dim=-1)

class KoPATrainer(Trainer):
    """
    自定义 Trainer：在 LM loss 上叠加 DDI 辅助损失。
    支持两种 DDI 监督来源：
      1) ddi_data: 需实现 sample_batch(batch_size, neg_ratio, device) -> (h_pos, t_pos, t_neg, w_pos, rid)
      2) subgraph_ddi: 需实现 propose_pairs(seed_drug_ints) -> (pairs, weights)
         这里的 seed_drug_ints 来自 batch 的 inputs['embedding_ids']（h/t 实体）
    """
    def __init__(self, *args,
                 ddi_data=None,
                 subgraph_ddi=None,
                 ddi_lambda: float = 0.0,
                 ddi_neg_ratio: int = 1,
                 drug_ints=None,             # set/list of drug entity int-ids (用于子图法)
                 freeze_kge: bool = True,    # 默认冻结 KGE 权重
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ddi_data = ddi_data
        self.subgraph_ddi = subgraph_ddi
        self.ddi_lambda = ddi_lambda
        self.ddi_neg_ratio = ddi_neg_ratio
        self.drug_ints = set(drug_ints) if drug_ints is not None else None

        # 获取模型所在设备
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 准备 DDI 关系向量（若 KoPAWithAdapter 未提供）
        if not hasattr(self.model, "ddi_rel") or self.model.ddi_rel is None:
            # 推断维度
            _, rel_w = _resolve_ent_rel_embeddings(self.model)
            rel_dim = rel_w.shape[1]
            # 在正确的设备上创建参数
            self.model.ddi_rel = nn.Parameter(torch.randn(rel_dim, device=model_device) * 0.02)
            self.model.register_parameter("ddi_rel", self.model.ddi_rel)
        else:
            # 如果已存在但在错误设备上，移动它
            if self.model.ddi_rel.device != model_device:
                self.model.ddi_rel = nn.Parameter(
                    self.model.ddi_rel.data.to(model_device),
                    requires_grad=self.model.ddi_rel.requires_grad
                )

        # 冻结/解冻 KGE 参数（按需）
        if freeze_kge:
            try:
                ent_w, rel_w = _resolve_ent_rel_embeddings(self.model)
                if hasattr(ent_w, "requires_grad_"): ent_w.requires_grad_(False)
                if hasattr(rel_w, "requires_grad_"): rel_w.requires_grad_(False)
            except Exception as e:
                print(f"[warn] freeze_kge failed: {e}")

    def _ddi_loss_from_dataset(self, device, per_device_bs):
        """从预先构建的 ddi_data 采样一个小批次计算 BCE 损失。"""
        if self.ddi_data is None or self.ddi_lambda <= 0:
            return torch.tensor(0.0, device=device)

        ent_w, rel_w = _resolve_ent_rel_embeddings(self.model)
        h_pos, t_pos, t_neg, w_pos, rid = self.ddi_data.sample_batch(
            batch_size=per_device_bs, neg_ratio=self.ddi_neg_ratio, device=device
        )
        r = rel_w[rid].unsqueeze(0) if isinstance(rid, int) else rel_w[rid].unsqueeze(0)
        s_pos = _distmult_score(ent_w[h_pos], r.expand_as(ent_w[h_pos]), ent_w[t_pos])

        h_rep = ent_w[h_pos].repeat_interleave(self.ddi_neg_ratio, dim=0)
        r_rep = r.expand(h_rep.shape[0], -1)
        s_neg = _distmult_score(h_rep, r_rep, ent_w[t_neg])

        y_pos = torch.ones_like(s_pos)
        y_neg = torch.zeros_like(s_neg)
        bce_pos = F.binary_cross_entropy_with_logits(s_pos, y_pos, weight=w_pos)
        bce_neg = F.binary_cross_entropy_with_logits(s_neg, y_neg)
        return bce_pos + bce_neg

    def _ddi_loss_from_subgraph(self, inputs, device, per_device_bs):
        """基于当前 batch 的 embedding_ids 在线生成子图药对并计算 BCE 损失。"""
        if self.subgraph_ddi is None or self.ddi_lambda <= 0 or "embedding_ids" not in inputs:
            return torch.tensor(0.0, device=device)
        if self.drug_ints is None:
            return torch.tensor(0.0, device=device)

        emb_ids = inputs["embedding_ids"]  # 期望 [B, 3] (h, r, t)
        if emb_ids.dim() != 2 or emb_ids.size(-1) < 3:
            return torch.tensor(0.0, device=device)

        seeds = torch.unique(torch.cat([emb_ids[:,0], emb_ids[:,2]], dim=0)).tolist()
        seed_drug_ints = [int(x) for x in seeds if int(x) in self.drug_ints]
        if not seed_drug_ints:
            return torch.tensor(0.0, device=device)

        pos_pairs, weights = self.subgraph_ddi.propose_pairs(seed_drug_ints)
        if len(pos_pairs) == 0:
            return torch.tensor(0.0, device=device)

        ent_w, rel_w = _resolve_ent_rel_embeddings(self.model)
        r = self.model.ddi_rel.unsqueeze(0)  # [1, D]

        pos_pairs = torch.tensor(pos_pairs, dtype=torch.long, device=device)   # [N, 2]
        w_pos     = torch.tensor(weights,   dtype=torch.float, device=device)  # [N]

        # negative sampling：随机替换尾部
        neg_count = max(1, self.ddi_neg_ratio * len(pos_pairs))
        drug_pool = torch.tensor(sorted(list(self.drug_ints)), dtype=torch.long, device=device)
        idx = torch.randint(high=drug_pool.numel(), size=(neg_count,), device=device)
        t_neg = drug_pool[idx]

        h_pos = ent_w[pos_pairs[:,0]]
        t_pos = ent_w[pos_pairs[:,1]]
        s_pos = _distmult_score(h_pos, r.expand_as(h_pos), t_pos)

        h_rep = h_pos.repeat_interleave(max(1,self.ddi_neg_ratio), dim=0)
        r_rep = r.expand(h_rep.shape[0], -1)
        s_neg = _distmult_score(h_rep, r_rep, ent_w[t_neg])

        y_pos = torch.ones_like(s_pos)
        y_neg = torch.zeros_like(s_neg)
        bce_pos = F.binary_cross_entropy_with_logits(s_pos, y_pos, weight=w_pos)
        bce_neg = F.binary_cross_entropy_with_logits(s_neg, y_neg)
        return bce_pos + bce_neg

    def compute_loss(self, model, inputs, return_outputs=False):
        # (1) 标准 LM 损失
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # 每设备 batch size
        per_bs = self.args.per_device_train_batch_size
        device = lm_loss.device

        # (2) DDI 辅助损失（优先使用预构建 dataset，其次在线子图）
        ddi_loss = torch.tensor(0.0, device=device)
        if self.ddi_lambda > 0:
            if self.ddi_data is not None:
                ddi_loss = ddi_loss + self._ddi_loss_from_dataset(device, per_bs)
            if self.subgraph_ddi is not None:
                ddi_loss = ddi_loss + self._ddi_loss_from_subgraph(inputs, device, per_bs)

        loss = lm_loss + self.ddi_lambda * ddi_loss
        return (loss, outputs) if return_outputs else loss
# ====== end of KoPATrainer ======

class KoPA(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM
    ) -> None:
        super(KoPA, self).__init__()
        self.llama_model = model
        # self.embeddings = nn.Embedding(100, 4096)
        self.embeddings = PrefixKGEmbedding(
            num_ent=2034,
            num_rel=42,
            dim_llm=4096,
            num_prefix=1
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None
    ):
        kg_embeds = self.embeddings(embedding_ids)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_mask = torch.ones((batch_size, seq_len))
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class KoPAWithAdapter(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_prefix: int,
        kge_model: str = "data/UMLS-rotate.pth",
        pretrain_emb_path = None
    ) -> None:
        super(KoPAWithAdapter, self).__init__()
        self.llama_model = model
        ent_embs, rel_embs = load_pretrain_kge(kge_model)

        if pretrain_emb_path is None:
            print("Adapter Trained From Scratch".format(pretrain_emb_path))
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=ent_embs,
                pretrain_rel_embs=rel_embs,
                dim_llm=4096,
                num_prefix=num_prefix
            )
        else:
            print("Adapter Load From {}".format(pretrain_emb_path))
            self.embeddings = torch.load(pretrain_emb_path)

        # === NEW: add a learnable DDI relation vector (for DistMult scoring) ===
        rel_dim = rel_embs.shape[1]                          ### NEW
        self.ddi_rel = nn.Parameter(torch.randn(rel_dim) * 0.02)  ### NEW

        # === 添加 Trainer 需要的属性 ===
        self._keys_to_ignore_on_save = None
        self.config = getattr(model, 'config', None)

        # === 添加 Trainer 需要的属性 ===
        self._keys_to_ignore_on_save = None
        self.config = getattr(model, 'config', None)

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None
    ):
        kg_embeds = self.embeddings(embedding_ids)
        # print(kg_embeds.shape)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_mask = torch.ones((batch_size, seq_len))
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # === NEW: expose entity/relation embedding weights to the trainer ===
    def get_entity_relation_weights(self):                          ### NEW
        embs = self.embeddings
        # 常见命名兼容：优先 nn.Embedding.weight；其次 dict['*.weight']
        ent_w = getattr(getattr(embs, 'ent_embeddings', None), 'weight', None)
        rel_w = getattr(getattr(embs, 'rel_embeddings', None), 'weight', None)
        if ent_w is None and isinstance(embs, dict):
            ent_w = embs.get('ent_embeddings.weight', None)
        if rel_w is None and isinstance(embs, dict):
            rel_w = embs.get('rel_embeddings.weight', None)
        if ent_w is None or rel_w is None:
            raise RuntimeError("Could not resolve KGE entity/relation weights from self.embeddings")
        return ent_w, rel_w                                               ### NEW



class PrefixKGEmbedding(nn.Module):
    def __init__(
        self,
        num_ent,
        num_rel,
        dim_llm,
        num_prefix
    ):
        super(PrefixKGEmbedding, self).__init__()
        self.emb_dim = num_prefix * dim_llm
        self.ent_embeddings = nn.Embedding(num_ent, self.emb_dim)
        self.rel_embeddings = nn.Embedding(num_rel, self.emb_dim)
    

    def forward(self, triple_ids):
        head, relation, tail = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2]
        h = self.ent_embeddings(head)
        r = self.rel_embeddings(relation)
        t = self.ent_embeddings(tail)
        prefix = torch.stack((h, r, t), dim=1)
        return prefix

class PretrainKGEmbedding(nn.Module):
    def __init__(
        self,
        pretrain_ent_embs,
        pretrain_rel_embs,
        dim_llm,
        num_prefix
    ):
        super(PretrainKGEmbedding, self).__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm
        self.emb_dim = num_prefix * dim_llm
        self.ent_embeddings = nn.Embedding.from_pretrained(pretrain_ent_embs)
        self.rel_embeddings = nn.Embedding.from_pretrained(pretrain_rel_embs)
        self.pretrain_dim = self.ent_embeddings.weight.shape[1]
        # Froze the pretrain embeddings
        self.ent_embeddings.requires_grad_(False)
        self.rel_embeddings.requires_grad_(False)
        self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)
    

    def forward(self, triple_ids):
        # main training stage
        if triple_ids.shape[1] == 3:
            head, relation, tail = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2]
            h = self.ent_embeddings(head)
            r = self.rel_embeddings(relation)
            t = self.ent_embeddings(tail)
            pretrain_embs = torch.stack((h, r, t), dim=1)
            prefix = self.adapter(pretrain_embs).reshape(-1, 3*self.num_prefix, self.llm_dim)
            return prefix
        # entity-aware pre-funing
        else:
            ent = triple_ids.reshape(-1,)
            emb = self.ent_embeddings(ent)
            prefix = self.adapter(emb).reshape(-1, self.num_prefix, self.llm_dim)
            # print(prefix.shape)
            return prefix

