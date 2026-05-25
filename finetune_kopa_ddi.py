import os
import sys
import math
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from kopa import KoPA, KoPAWithAdapter

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from kopaddi import KoPATrainer
from utils.prompter import Prompter
import json, random, pandas as pd
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import Dataset


# ========== TSVDDIData: 从 TSV 文件加载 DDI 正样本对 ==========
class TSVDDIData:
    """
    从 TSV 文件加载 DDI 正样本对，用于 KoPATrainer 的 ddi_data 参数。
    TSV 格式: emb_drug1\temb_drug2 (两列，都是 entity embedding id)
    """
    def __init__(self, pos_tsv_path: str, drug_pool=None, rid: int = None):
        pairs = []
        with open(pos_tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    a, b = int(parts[0]), int(parts[1])
                    pairs.append((a, b))
        assert len(pairs) > 0, f"Empty DDI file: {pos_tsv_path}"
        print(f"[TSVDDIData] 加载了 {len(pairs)} 个 DDI 正样本对")

        self.pairs = torch.tensor(pairs, dtype=torch.long)   # [N,2]
        self.rid = rid

        # 默认 drug_pool = DDI pairs 中出现过的所有 drug embedding id
        uniq_from_pairs = sorted(set([x for ab in pairs for x in ab]))
        if drug_pool is None or len(drug_pool) == 0:
            self.drug_pool = torch.tensor(uniq_from_pairs, dtype=torch.long)
        else:
            # 合并 drug_pool 和 pairs 中的 drug
            all_drugs = set(uniq_from_pairs) | set(drug_pool)
            self.drug_pool = torch.tensor(sorted(list(all_drugs)), dtype=torch.long)
        
        print(f"[TSVDDIData] drug_pool 大小: {len(self.drug_pool)}")

        # 用于避免采到正样本作为负样本
        self.pos_set = set(pairs) | set([(b, a) for a, b in pairs])

    def sample_batch(self, batch_size: int, neg_ratio: int, device):
        n = self.pairs.size(0)
        idx = torch.randint(high=n, size=(batch_size,))
        pos = self.pairs[idx]  # [B,2]
        h_pos = pos[:, 0].to(device)
        t_pos = pos[:, 1].to(device)

        # 权重先全 1
        w_pos = torch.ones(batch_size, dtype=torch.float, device=device)

        # 确保 drug_pool 在正确设备上且不为空
        pool = self.drug_pool.to(device)
        pool_size = pool.numel()
        
        if pool_size == 0:
            # 如果 pool 为空，使用 pairs 中的所有 drug 作为 pool
            all_drugs = torch.unique(self.pairs.flatten()).to(device)
            pool = all_drugs
            pool_size = pool.numel()
        
        # 负采样：对每个 h 采 neg_ratio 个 t_neg
        t_negs = []
        for i in range(batch_size * neg_ratio):
            h = int(h_pos[i // neg_ratio].item())
            # 采到不是正样本为止（最多尝试几次避免死循环）
            for _ in range(10):
                rand_idx = torch.randint(low=0, high=pool_size, size=(1,)).item()
                cand = int(pool[rand_idx].item())
                if (h, cand) not in self.pos_set:
                    t_negs.append(cand)
                    break
            else:
                t_negs.append(cand)  # 兜底
        t_neg = torch.tensor(t_negs, dtype=torch.long, device=device)

        # rid：如果你想用 relation embedding 的 "drug_drug" 那个向量，就返回 rid
        rid = self.rid if self.rid is not None else 0
        return h_pos, t_pos, t_neg, w_pos, rid
# ========== END TSVDDIData ==========


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_kg_indices(kg_tsv: str):
    """
    读取训练用KG边，构建子图查询索引：
    - disease -> set(drug)
    - drug    -> set(disease)  (TREAT)
    - drug    -> set(adr)      (DRUG_SIDER / SIDER_DRUG，可选)
    """
    if not os.path.exists(kg_tsv):
        raise FileNotFoundError(f"kg_tsv not found: {kg_tsv}")
    df = pd.read_csv(
        kg_tsv, sep="\t", header=None, names=["h", "r", "t"], dtype=str, compression="infer"
    )
    dis2drugs = defaultdict(set)
    drug2dis = defaultdict(set)
    drug2adr = defaultdict(set)
    has_sider = False
    for h, r, t in df.itertuples(index=False):
        if r == "TREAT" and isinstance(h, str) and h.startswith("DB"):
            drug2dis[h].add(t)
            dis2drugs[t].add(h)
        elif r in ("DRUG_SIDER", "SIDER_DRUG"):
            has_sider = True
            drug, adr = (h, t) if r == "DRUG_SIDER" else (t, h)
            if isinstance(drug, str) and drug.startswith("DB"):
                drug2adr[drug].add(adr)
    return dict(dis2drugs=dis2drugs, drug2dis=drug2dis, drug2adr=drug2adr, has_sider=has_sider)


def build_id_maps(entity2id_path: str):
    e2i = load_json(entity2id_path)
    i2e = {v: k for k, v in e2i.items()}
    drug_ints = {v for k, v in e2i.items() if isinstance(k, str) and k.startswith("DB")}
    return e2i, i2e, drug_ints


def distmult_score(h, r, t):
    # h, r, t: [B, D] -> [B]
    return (h * r * t).sum(dim=-1)


class SubgraphDDIRegularizer:
    def __init__(
        self,
        indices,
        e2i,
        i2e,
        drug_ints,
        min_shared=1,
        jaccard_tau=0.0,
        max_degree_cap=500,
    ):
        self.dis2drugs = indices["dis2drugs"]
        self.drug2dis = indices["drug2dis"]
        self.drug2adr = indices["drug2adr"]
        self.has_sider = indices["has_sider"]
        self.e2i, self.i2e = e2i, i2e
        self.drug_ints = drug_ints
        self.min_shared = min_shared
        self.jaccard_tau = jaccard_tau
        self.max_degree_cap = max_degree_cap

    def _cotreat_pairs_from_seeds(self, seeds_ext):
        # B1: 以 seeds 为中心的 2-hop 子图: drug->disease<-drug
        # 收集所有疾病，再汇总这些疾病下的药物；按疾病共现计数
        from collections import Counter

        counter = Counter()
        pool = set()
        for d in seeds_ext:
            for dis in list(self.drug2dis.get(d, [])):
                drugs = sorted(self.dis2drugs.get(dis, []))
                if len(drugs) > self.max_degree_cap:
                    drugs = drugs[: self.max_degree_cap]
                pool.update(drugs)
        pool = sorted(pool)

        dis_of = {dr: self.drug2dis.get(dr, set()) for dr in pool}
        for i in range(len(pool)):
            A = dis_of[pool[i]]
            for j in range(i + 1, len(pool)):
                inter = A & dis_of[pool[j]]
                if len(inter) >= self.min_shared:
                    counter[tuple(sorted((pool[i], pool[j])))] += len(inter)
        pos = [(a, b, float(w)) for (a, b), w in counter.items()]
        return pos  # [(drug_ext_a, drug_ext_b, weight), ...]

    def _sider_pairs_from_pool(self, pool_ext):
        # B2: 共享 ADR；若没有 SIDER 则返回空
        if not self.has_sider:
            return []
        pool = sorted(pool_ext)
        adrs = {dr: self.drug2adr.get(dr, set()) for dr in pool}
        rows = []
        for i in range(len(pool)):
            A = adrs[pool[i]]
            for j in range(i + 1, len(pool)):
                B = adrs[pool[j]]
                inter = A & B
                if len(inter) == 0:
                    continue
                if self.jaccard_tau > 0:
                    denom = len(A | B)
                    jac = (len(inter) / denom) if denom > 0 else 0.0
                    if jac >= self.jaccard_tau:
                        rows.append((pool[i], pool[j], float(jac)))
                else:
                    rows.append((pool[i], pool[j], float(len(inter))))
        return rows

    def propose_pairs(self, seed_drug_ints):
        """seed_drug_ints: 一批样本里出现的药物（int ids）"""
        # 转回外部ID便于查表
        seeds_ext = [self.i2e[i] for i in seed_drug_ints if i in self.i2e]
        # B1: 从 seeds 扩展的子图内生成共治药对
        b1 = self._cotreat_pairs_from_seeds(seeds_ext)  # list of (ext, ext, w)
        # 构建 pool（b1 里所有药 + 种子药）
        pool_ext = set(seeds_ext)
        for a, b, _ in b1:
            pool_ext.add(a)
            pool_ext.add(b)
        # B2: 在这个 pool 内再用 SIDER 做共享 ADR
        b2 = self._sider_pairs_from_pool(pool_ext)
        # 合并去重（权重相加）
        from collections import defaultdict

        acc = defaultdict(float)
        for a, b, w in b1:
            key = tuple(sorted((a, b)))
            acc[key] += w
        for a, b, w in b2:
            key = tuple(sorted((a, b)))
            acc[key] += w
        # 转回 int id；过滤非药物
        pos_pairs = []
        weights = []
        for (a, b), w in acc.items():
            ia, ib = self.e2i.get(a), self.e2i.get(b)
            if ia is None or ib is None:
                continue
            if ia in self.drug_ints and ib in self.drug_ints and ia != ib:
                pos_pairs.append((ia, ib))
                weights.append(w)
        return pos_pairs, weights


def resolve_ent_rel_embeddings(kopa_model):
    """
    从 KoPAWithAdapter 里拿到实体/关系嵌入矩阵（权重张量）。
    兼容常见字段命名：ent_embeddings(.weight)/rel_embeddings(.weight)
    """
    embs = getattr(kopa_model, "embeddings", None)
    if embs is None:
        raise RuntimeError("KoPAWithAdapter 需要暴露 .embeddings 才能取到KGE矩阵")
    # 可能是 nn.Embedding 或 dict[tensor]
    ent_w = getattr(embs, "ent_embeddings", None)
    if ent_w is not None and hasattr(ent_w, "weight"):
        ent_w = ent_w.weight
    elif isinstance(embs, dict):
        ent_w = embs.get("ent_embeddings.weight", None)
    rel_w = getattr(embs, "rel_embeddings", None)
    if rel_w is not None and hasattr(rel_w, "weight"):
        rel_w = rel_w.weight
    elif isinstance(embs, dict):
        rel_w = embs.get("rel_embeddings.weight", None)
    if ent_w is None or rel_w is None:
        raise RuntimeError("未能解析实体/关系嵌入")
    return ent_w, rel_w


# ---- Collator that guarantees embedding_ids are int64 (and batched) ----
from transformers import DataCollatorForSeq2Seq
class DataCollatorWithKG(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)

        ids_list  = []
        mask_list = []

        for f in features:
            # if a sample has no embedding_ids at all, create a safe dummy [0] with mask [0]
            if "embedding_ids" not in f or f["embedding_ids"] is None:
                e = torch.tensor([0], dtype=torch.long)
                m = torch.tensor([0], dtype=torch.long)
            else:
                e = f["embedding_ids"]
                if not torch.is_tensor(e):
                    e = torch.tensor(e, dtype=torch.long)
                else:
                    e = e.to(torch.long)

                # if no mask was provided, default to all ones (real ids)
                if "embedding_mask" in f and f["embedding_mask"] is not None:
                    m = f["embedding_mask"]
                    if not torch.is_tensor(m):
                        m = torch.tensor(m, dtype=torch.long)
                    else:
                        m = m.to(torch.long)
                else:
                    m = torch.ones_like(e, dtype=torch.long)

                # special-case: if this was a dummy/everything padded later, you can keep it as ones;
                # the model should still zero out by attention or your own logic.

            ids_list.append(e)
            mask_list.append(m)

        # pad to [B, L]
        ids  = torch.nn.utils.rnn.pad_sequence(ids_list,  batch_first=True, padding_value=0)
        mask = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=0)

        batch["embedding_ids"]  = ids
        batch["embedding_mask"] = mask
        return batch


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "YOUR LLM PATH",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 16,
    micro_batch_size: int = 16,
    num_epochs: int = 2,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    num_prefix: int = 1,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    kge_model: str = "data/CoDeX-S.pth",
    kg_tsv: str = "",  # 训练用KG边(三列: h r t)，仅训练集，避免泄漏
    entity2id_path: str = "",  # string -> int
    relation2id_path: str = "",  # relation -> int（可选，用于判断关系名是否存在）
    ddi_lambda: float = 0.5,  # DDI 辅助损失权重
    ddi_neg_ratio: int = 1,  # 每个正样本的负样本倍数
    ddi_pos_path: str = "",  # DDI 正样本对文件路径 (TSV: emb_drug1\temb_drug2)
    min_shared: int = 1,  # B1: 至少共同治疗多少疾病算伪正
    jaccard_tau: float = 0.0,  # B2: 无严重列表时的 Jaccard 阈值
    max_degree_cap: int = 500,  # B1: 单疾病最多取多少药（控复杂度）
    allow_update_kge: bool = False,
):
    # --- DDP world info & safe grad accumulation ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    safe_gas = max(1, math.ceil(batch_size / (micro_batch_size * max(1, world_size))))

    # ===== 关键修复：DDP 设备设置 =====
    if torch.cuda.is_available():
        # 每个进程绑定到自己的 local_rank 对应的 GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if local_rank == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"num_prefix: {num_prefix}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"kge model: {kge_model}\n"
            f"[DDP] world_size={world_size}, local_rank={local_rank}, grad_accum={safe_gas}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    # LLM + tokenizer
    # ===== 关键修复：不使用 device_map，让模型先加载到 CPU，然后手动移到正确的 GPU =====
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=None,  # 不自动分配设备
        low_cpu_mem_usage=True,
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    prompter = Prompter(prompt_template_name)
    rel2i = load_json(relation2id_path) if relation2id_path else None

    # --- build KG maps BEFORE dataset.map so generate_and_tokenize_prompt can use them ---
    assert kg_tsv, "kg_tsv 不能为空（建议用train.tsv，三列 h r t）"
    assert entity2id_path, "entity2id_path 不能为空（JSON: external_id -> int_id）"
    indices = build_kg_indices(kg_tsv)  # disease->drugs, drug->disease, drug->ADR
    e2i, i2e, drug_ints = build_id_maps(entity2id_path)  # 拿到药物实体ID集合

    def tokenize(prompt, add_eos_token=True):
        # tokenize with optional EOS padding
        result = tokenizer(
            prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None
        )
        if (
            result["input_ids"]
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    # ensures embedding_ids are LONG at source
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point.get("input", ""), data_point.get("output", "")
        )
        tok = tokenize(full_prompt)

        # 1) 如果样本自带 embedding_ids，直接用（强制 long）
        if "embedding_ids" in data_point:
            ids = data_point["embedding_ids"]
            tok["embedding_ids"]  = torch.tensor(ids, dtype=torch.long)
            tok["embedding_mask"] = torch.ones(len(ids), dtype=torch.long)
            return tok

        # 2) 若样本提供外部ID，转成整型ID
        h_ext = data_point.get("h")
        r_name = data_point.get("r")
        t_ext = data_point.get("t")
        if h_ext is not None and t_ext is not None and h_ext in e2i and t_ext in e2i:
            r_id = 0
            if rel2i and r_name in rel2i:
                r_id = int(rel2i[r_name])
            ids = [int(e2i[h_ext]), int(r_id), int(e2i[t_ext])]
            tok["embedding_ids"]  = torch.tensor(ids, dtype=torch.long)
            tok["embedding_mask"] = torch.ones(len(ids), dtype=torch.long)
            return tok

        tok["embedding_ids"]  = torch.tensor([0], dtype=torch.long)
        tok["embedding_mask"] = torch.tensor([0], dtype=torch.long)
        return tok

    # LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    # ===== 关键修复：先将 PEFT 模型移到正确的 GPU =====
    model = model.to(device)
    
    # 创建 KoPAWithAdapter，此时 model 已经在正确的 GPU 上
    slama_model = KoPAWithAdapter(model, num_prefix, kge_model=kge_model)
    
    # ===== 关键修复：使用 cuda() 方法直接移动整个模型 =====
    # 这比手动遍历更可靠
    slama_model = slama_model.to(device)
    
    # 再次确认：遍历所有子模块并移动
    for name, module in slama_model.named_modules():
        module.to(device)
    
    # 详细检查
    def check_model_devices(model, expected_device, rank):
        cpu_params = []
        cpu_buffers = []
        
        for name, param in model.named_parameters():
            if param.device.type == 'cpu':
                cpu_params.append(name)
        
        for name, buf in model.named_buffers():
            if buf.device.type == 'cpu':
                cpu_buffers.append(name)
        
        if cpu_params:
            print(f"[rank{rank}][ERROR] CPU 上的参数: {cpu_params}")
        if cpu_buffers:
            print(f"[rank{rank}][ERROR] CPU 上的 buffers: {cpu_buffers}")
        
        if not cpu_params and not cpu_buffers:
            print(f"[rank{rank}][OK] 所有参数和 buffers 都在 {expected_device} 上")
            return True
        return False
    
    check_model_devices(slama_model, device, local_rank)

    # 数据集
    if data_path.endswith((".json", ".jsonl")):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # full
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")  # LoRA
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    if torch.cuda.device_count() > 1:
        slama_model.is_parallelizable = True
        slama_model.model_parallel = True
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    subgraph_obj = SubgraphDDIRegularizer(
        indices,
        e2i,
        i2e,
        drug_ints,
        min_shared=min_shared,
        jaccard_tau=jaccard_tau,
        max_degree_cap=max_degree_cap,
    )
    drug_id_set = list(drug_ints)

    data_collator = DataCollatorWithKG(
        tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    # ========== 构建 DDI 数据 ==========
    ddi_data = None
    if ddi_lambda > 0 and ddi_pos_path:
        # 从 TSV 文件加载真实 DDI 正样本对
        print(f"[DDI] 从 {ddi_pos_path} 加载 DDI 正样本对...")
        
        # 尝试获取 drug_drug 关系的 rid
        ddi_rid = None
        if rel2i and isinstance(rel2i, dict):
            # 尝试不同的关系名
            for rel_name in ["drug_drug", "DDI", "ddi", "DRUG_DRUG"]:
                if rel_name in rel2i:
                    ddi_rid = int(rel2i[rel_name])
                    print(f"[DDI] 使用关系 '{rel_name}' (rid={ddi_rid})")
                    break
        
        ddi_data = TSVDDIData(
            pos_tsv_path=ddi_pos_path,
            drug_pool=drug_id_set,
            rid=ddi_rid
        )
        print(f"[DDI] DDI 数据加载完成，使用真实 DDI 对进行训练")
    elif ddi_lambda > 0:
        print(f"[DDI] 未提供 ddi_pos_path，将使用 subgraph_ddi 进行在线采样")

    trainer = KoPATrainer(
        model=slama_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        # DDI 相关参数
        ddi_data=ddi_data,  # 真实 DDI 正样本对（优先使用）
        subgraph_ddi=subgraph_obj if ddi_data is None else None,  # 如果有真实 DDI 数据，关闭 subgraph
        drug_ints=drug_id_set if ddi_data is None else None,  # subgraph 需要
        ddi_lambda=ddi_lambda,  # DDI 辅助 loss 权重
        ddi_neg_ratio=ddi_neg_ratio,  # 每个正样本的负样本倍数
        freeze_kge=True,  # 先冻结 KGE 更稳（想联合训练可设 False）

        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=safe_gas,  # SAFE & DDP-AWARE
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_hf",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=5000,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            run_name=None,
            dataloader_pin_memory=False,
        ),
        data_collator=data_collator,
    )
    
    # ===== 修复 KoPATrainer 中添加的 ddi_rel 参数 =====
    # KoPATrainer 可能在 model 上添加了额外的参数，需要移到正确设备
    if hasattr(trainer.model, 'ddi_rel') and trainer.model.ddi_rel is not None:
        if trainer.model.ddi_rel.device != device:
            trainer.model.ddi_rel = torch.nn.Parameter(
                trainer.model.ddi_rel.data.to(device),
                requires_grad=trainer.model.ddi_rel.requires_grad
            )
            print(f"[rank{local_rank}] 已将 ddi_rel 移到 {device}")
    model.config.use_cache = False

    # make PEFT state_dict saving work
    old_state_dict = model.state_dict

    model.state_dict = (  # type: ignore
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # 禁用 torch.compile - 它可能导致 DDP 问题
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # 在训练前再次检查设备状态
    print(f"[rank{local_rank}] 训练前最终检查...")
    check_model_devices(trainer.model, device, local_rank)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    torch.save(slama_model.embeddings, os.path.join(output_dir, "embeddings.pth"))

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
