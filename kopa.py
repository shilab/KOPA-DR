import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from transformers import LlamaForCausalLM
from process_kge import load_pretrain_kge


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
        embedding_ids: Optional[torch.LongTensor] = None,
        embedding_mask: Optional[torch.Tensor] = None,   # NEW
        **kwargs,                                        # swallow extra keys safely
    ):
        # Token embeddings
        if inputs_embeds is not None:
            token_embeds = inputs_embeds                        # [B, Lt, H]
        else:
            token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        device = token_embeds.device
        B, Lt, H = token_embeds.shape

        # KG embeddings (robust)
        if embedding_ids is None:
            kg_embeds = token_embeds.new_zeros((B, 0, H))
            kg_attn  = token_embeds.new_zeros((B, 0), dtype=torch.long)
            Lkg = 0
        else:
            try:
                emb_dev = next(self.embeddings.parameters()).device  # if nn.Module
            except Exception:
                emb_dev = device
            ids = embedding_ids.to(device=emb_dev, dtype=torch.long)
            kg = self.embeddings(ids)                 # [B, Lkg, H] on emb_dev
            kg_embeds = kg.to(device)                 # align to token device
            Lkg = kg_embeds.size(1)

            if embedding_mask is None:
                kg_attn = torch.ones((B, Lkg), dtype=torch.long, device=device)
            else:
                kg_attn = embedding_mask.to(device=device, dtype=torch.long)

            kg_embeds = kg_embeds * kg_attn.unsqueeze(-1)  # zero-out masked

        # Concat KG prefix + tokens
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)  # [B, Lkg+Lt, H]

        # Attention mask & labels
        if attention_mask is None:
            attention_mask = torch.ones((B, Lt), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
        new_attention_mask = torch.cat((kg_attn, attention_mask), dim=1)

        if labels is None:
            labels = torch.full((B, Lt), fill_value=-100, dtype=torch.long, device=device)
        else:
            labels = labels.to(device=device, dtype=torch.long)
        prefix_labels = torch.full((B, Lkg), fill_value=-100, dtype=torch.long, device=device)
        new_labels = torch.cat((prefix_labels, labels), dim=1)

        # Let backbone infer positions after length change
        position_ids = None

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
        self.num_prefix = num_prefix  # 保存以便后续使用
        
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
        
        # 注意：不在这里移动设备，让外部代码用 .to(device) 统一处理
    
    def forward2(
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
        embedding_ids: Optional[torch.LongTensor] = None,
        embedding_mask: Optional[torch.Tensor] = None,  # optional; 1=keep, 0=ignore
        **kwargs,
    ):
    # ===== 1) Token embeddings (put inputs on the SAME device as the token embedding weight) =====
        tok_weight = self.llama_model.model.model.embed_tokens.weight
        tok_dev = tok_weight.device

        if inputs_embeds is not None:
            token_embeds = inputs_embeds.to(device=tok_dev)
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            token_embeds = self.llama_model.model.model.embed_tokens(input_ids.to(device=tok_dev))

        B, Lt, H = token_embeds.shape

    # ===== 2) KG prefix embeddings (robust if missing) =====
        if embedding_ids is None:
            # No KG in this batch
            kg_embeds = token_embeds.new_zeros((B, 0, H))            # [B, 0, H]
            kg_attn  = token_embeds.new_zeros((B, 0), dtype=torch.long)
            Lkg = 0
        else:
            # put ids on the embeddings module's device (if it exists), then move result to token device
            try:
                emb_dev = next(self.embeddings.parameters()).device
            except Exception:
                emb_dev = tok_dev
            ids = embedding_ids.to(device=emb_dev, dtype=torch.long)

            kg = self.embeddings(ids)                                # expected [B, Lkg, H] on emb_dev
            kg_embeds = kg.to(tok_dev)
            Lkg = kg_embeds.size(1)

            # build/convert mask on token device
            if embedding_mask is None:
                kg_attn = torch.ones((B, Lkg), dtype=torch.long, device=tok_dev)
            else:
                kg_attn = embedding_mask.to(device=tok_dev, dtype=torch.long)
                # If caller ever sends shape [B,] (rare), expand safely
                if kg_attn.dim() == 1 and Lkg > 0:
                    kg_attn = kg_attn.unsqueeze(1).expand(B, Lkg)

         # zero-out masked KG slots (safe if Lkg==0 too)
            if Lkg > 0:
                kg_embeds = kg_embeds * kg_attn.unsqueeze(-1)

        # ===== 3) Concat KG prefix + tokens =====
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)   # [B, Lkg+Lt, H]

        # ===== 4) Attention mask & labels on correct device/dtype =====
        if attention_mask is None:
            attention_mask = torch.ones((B, Lt), dtype=torch.long, device=tok_dev)
        else:
            attention_mask = attention_mask.to(device=tok_dev, dtype=torch.long)

        new_attention_mask = torch.cat((kg_attn, attention_mask), dim=1)  # [B, Lkg+Lt]

        if labels is None:
            labels = torch.full((B, Lt), fill_value=-100, dtype=torch.long, device=tok_dev)
        else:
            labels = labels.to(device=tok_dev, dtype=torch.long)

        prefix_labels = torch.full((B, Lkg), fill_value=-100, dtype=torch.long, device=tok_dev)
        new_labels = torch.cat((prefix_labels, labels), dim=1)            # [B, Lkg+Lt]

    # We changed sequence length; let the backbone infer positions
        position_ids = None

    # ===== 5) Delegate to backbone =====
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

    def forward1(
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
        embedding_ids: Optional[torch.LongTensor] = None,
        embedding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # ---- 1) token embeddings on their native device ----
        if inputs_embeds is not None:
            token_embeds = inputs_embeds                      # [B, Lt, H]
        else:
            token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        device = token_embeds.device
        B, Lt, H = token_embeds.shape

        # ---- 2) KG prefix (robust to None/empty/mismatched shapes) ----
        # default: no KG prefix
        kg_embeds = token_embeds.new_zeros((B, 0, H))
        kg_attn   = token_embeds.new_zeros((B, 0), dtype=torch.long)
        Lkg = 0

        if (embedding_ids is not None) and (not (torch.is_tensor(embedding_ids) and embedding_ids.numel() == 0)):
            # put ids on the device of the embedding table
            try:
                emb_dev = next(self.embeddings.parameters()).device
            except Exception:
                emb_dev = device

            ids = torch.as_tensor(embedding_ids, dtype=torch.long, device=emb_dev)

            # normalize ids to [B, K]
            if ids.dim() == 1:                      # [K]
                ids = ids.unsqueeze(0).expand(B, -1)
            elif ids.dim() == 2:
                if ids.size(0) != B:
                    if ids.size(0) == 1:
                        ids = ids.expand(B, -1)
                    else:
                        raise ValueError(f"embedding_ids batch mismatch: got {ids.size(0)}, need {B}")
            else:
                raise ValueError(f"embedding_ids must be [K] or [B,K], got shape {tuple(ids.shape)}")

            # fetch embeddings -> [B, K, H] on emb_dev
            kg = self.embeddings(ids)
            if kg.dim() == 2:                       # [K, H] -> [B, K, H]
                kg = kg.unsqueeze(0).expand(B, -1, -1)

            # move to token device/dtype
            kg_embeds = kg.to(device=device, dtype=token_embeds.dtype)
            Lkg = kg_embeds.size(1)

            # build/correct mask to [B, K]
            if embedding_mask is None or (torch.is_tensor(embedding_mask) and embedding_mask.numel() == 0):
                kg_attn = torch.ones((B, Lkg), dtype=torch.long, device=device)
            else:
                m = torch.as_tensor(embedding_mask, dtype=torch.long)
                if m.dim() == 1:                    # [K?] -> [B, K?]
                    m = m.unsqueeze(0).expand(B, -1)
                elif m.dim() == 2 and m.size(0) != B:
                    if m.size(0) == 1:
                        m = m.expand(B, -1)
                    else:
                        raise ValueError(f"embedding_mask batch mismatch: got {m.size(0)}, need {B}")

            # crop/pad to K
                if m.size(1) > Lkg:
                    m = m[:, :Lkg]
                elif m.size(1) < Lkg:
                    pad = torch.zeros((B, Lkg - m.size(1)), dtype=m.dtype)
                    m = torch.cat([m, pad], dim=1)

                kg_attn = m.to(device=device, dtype=torch.long)

        # zero-out masked prefix positions
            kg_embeds = kg_embeds * kg_attn.unsqueeze(-1)

    # ---- 3) concat KG prefix + tokens ----
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)              # [B, Lkg+Lt, H]

    # ---- 4) attention mask & labels on correct device/dtypes ----
        if attention_mask is None:
            attention_mask = torch.ones((B, Lt), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
        new_attention_mask = torch.cat((kg_attn, attention_mask), dim=1)        # [B, Lkg+Lt]

        if labels is None:
            labels = torch.full((B, Lt), fill_value=-100, dtype=torch.long, device=device)
        else:
            labels = labels.to(device=device, dtype=torch.long)
        prefix_labels = torch.full((B, Lkg), fill_value=-100, dtype=torch.long, device=device)
        new_labels = torch.cat((prefix_labels, labels), dim=1)                  # [B, Lkg+Lt]

    # Let backbone infer positions after we changed sequence length
        position_ids = None

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

    def ffforward(
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
        embedding_ids: Optional[torch.Tensor] = None,   # [B, K] or [K] or None
        embedding_mask: Optional[torch.Tensor] = None,  # [B, K] or [K] or None; 1=keep,0=pad
        **kwargs,
    ):
        # -- 0) pick the canonical device from the token embedding weight
        tok_weight = self.llama_model.model.model.embed_tokens.weight
        tok_dev = tok_weight.device

        # -- 1) token embeddings (ensure they are on tok_dev)
        if inputs_embeds is not None:
            token_embeds = inputs_embeds.to(tok_dev)
            B, Lt, H = token_embeds.shape
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            input_ids = input_ids.to(tok_dev)
            token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
            B, Lt, H = token_embeds.shape

        # -- 2) build KG prefix (robust to missing ids/mask)
        # normalize embedding_ids -> [B, K] of dtype long
        kg_len = 0
        if embedding_ids is None:
            # no prefix
            kg_embeds = torch.zeros((B, 0, H), dtype=token_embeds.dtype, device=tok_dev)
            kg_attn   = torch.zeros((B, 0),    dtype=torch.long,        device=tok_dev)
        else:
            if embedding_ids.dim() == 1:
                # [K] -> [B, K] (broadcast same prefix to all in batch)
                embedding_ids = embedding_ids.unsqueeze(0).expand(B, -1)
            elif embedding_ids.dim() == 2 and embedding_ids.size(0) != B:
                # if single row given for a batch, expand; otherwise enforce same B
                if embedding_ids.size(0) == 1:
                    embedding_ids = embedding_ids.expand(B, -1)
                else:
                    raise ValueError(f"embedding_ids batch mismatch: got {embedding_ids.size(0)}, need {B}")
            embedding_ids = embedding_ids.to(tok_dev, dtype=torch.long)

            # mask handling
            if embedding_mask is None:
                kg_attn = torch.ones_like(embedding_ids, dtype=torch.long, device=tok_dev)
            else:
                if embedding_mask.dim() == 1:
                    embedding_mask = embedding_mask.unsqueeze(0).expand(B, -1)
                elif embedding_mask.dim() == 2 and embedding_mask.size(0) != B:
                    if embedding_mask.size(0) == 1:
                        embedding_mask = embedding_mask.expand(B, -1)
                    else:
                        raise ValueError(f"embedding_mask batch mismatch: got {embedding_mask.size(0)}, need {B}")
                kg_attn = embedding_mask.to(tok_dev, dtype=torch.long)

            # compute KG embeddings; your .embeddings should accept [B, K] long ids and return [B, K, H]
            kg_embeds = self.embeddings(embedding_ids)  # expect float on some device
            kg_embeds = kg_embeds.to(tok_dev, dtype=token_embeds.dtype)

            # apply mask (zero-out padded slots) and compute actual prefix length (for labels later)
            kg_embeds = kg_embeds * kg_attn.unsqueeze(-1)  # [B, K, H] * [B, K, 1]
            kg_len = kg_embeds.size(1)

        # -- 3) concat prefix + tokens
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)  # [B, K+Lt, H]

        # -- 4) attention mask (same device & dtype long)
        if attention_mask is None:
            attention_mask = torch.ones((B, Lt), dtype=torch.long, device=tok_dev)
        else:
            attention_mask = attention_mask.to(tok_dev, dtype=torch.long)
        new_attention_mask = torch.cat((kg_attn, attention_mask), dim=1)  # [B, K+Lt]

        # -- 5) labels (prefix = -100 so it doesn't contribute to LM loss)
        if labels is None:
            labels = torch.full((B, Lt), fill_value=-100, dtype=torch.long, device=tok_dev)
        else:
            labels = labels.to(tok_dev, dtype=torch.long)
        if kg_len > 0:
            prefix_labels = torch.full((B, kg_len), fill_value=-100, dtype=torch.long, device=tok_dev)
            new_labels = torch.cat((prefix_labels, labels), dim=1)
        else:
            new_labels = labels

        # When changing sequence length via inputs_embeds, let backbone handle positions
        position_ids = None

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

    def forward3(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        embedding_ids=None,
        embedding_mask=None,
        **kwargs,
    ):
        # 1) Decide token-embedding device from the model itself
        tok_weight = self.llama_model.model.model.embed_tokens.weight
        tok_dev = tok_weight.device  # >>> key line

        # 2) Build token_embeds on the SAME device as embed_tokens.weight
        if inputs_embeds is not None:
            token_embeds = inputs_embeds.to(tok_dev)                  # >>>
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            input_ids = input_ids.to(tok_dev)                         # >>>
            token_embeds = self.llama_model.model.model.embed_tokens(input_ids)

        B, Lt, H = token_embeds.shape

        # 3) (Your existing normalization for embedding_ids/mask goes here)
        #    Make sure the computed kg_embeds and kg_attn end up on tok_dev.
        #    Example final lines after you’ve built them:
        kg_embeds = kg_embeds.to(tok_dev)                             # >>>
        kg_attn   = kg_attn.to(tok_dev)                               # >>>
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)    # [B, Lkg+Lt, H]

        # 4) Move masks/labels to the SAME device as input_embeds
        if attention_mask is None:
            attention_mask = torch.ones((B, Lt), dtype=torch.long, device=tok_dev)
        else:
            attention_mask = attention_mask.to(tok_dev, dtype=torch.long)  # >>>
        new_attention_mask = torch.cat((kg_attn, attention_mask), dim=1)

        if labels is None:
            labels = torch.full((B, Lt), fill_value=-100, dtype=torch.long, device=tok_dev)
        else:
            labels = labels.to(tok_dev, dtype=torch.long)             # >>>
        prefix_labels = torch.full((kg_attn.size(0), kg_attn.size(1)), fill_value=-100, dtype=torch.long, device=tok_dev)
        new_labels = torch.cat((prefix_labels, labels), dim=1)

        position_ids = None  # let the backbone infer due to length change

        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,   # lives on tok_dev
            labels=new_labels,            # lives on tok_dev
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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
        embedding_ids: Optional[torch.LongTensor] = None,
        embedding_mask: Optional[torch.Tensor] = None,   # NEW
        **kwargs,                                        # absorb extras safely
    ):
        # --- token embeddings ---
        if inputs_embeds is not None:
            token_embeds = inputs_embeds                                    # [B, Lt, H]
        else:
            token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        device = token_embeds.device
        B, Lt, H = token_embeds.shape

        # --- KG embeddings (robust even if missing) ---
        if embedding_ids is None:
            # no KG prefix in this batch
            kg_embeds = token_embeds.new_zeros((B, 0, H))
            kg_attn  = token_embeds.new_zeros((B, 0), dtype=torch.long)
            Lkg = 0
        else:
            # put ids on the embedding table device (if it's an nn.Module)
            try:
                emb_dev = next(self.embeddings.parameters()).device
            except Exception:
                emb_dev = device
            ids = embedding_ids.to(device=emb_dev, dtype=torch.long)

            # self.embeddings(...) must return [B, Lkg, H]
            kg = self.embeddings(ids)                                       # [B, Lkg, H] on emb_dev
            kg_embeds = kg.to(device)                                       # move to token device
            Lkg = kg_embeds.size(1)

            # build/convert mask: 1=keep, 0=ignore
            if embedding_mask is None:
                kg_attn = torch.ones((B, Lkg), dtype=torch.long, device=device)
            else:
                kg_attn = embedding_mask.to(device=device, dtype=torch.long)

            # zero-out masked positions (optional but safe)
            # Handle case where kg_attn has fewer entries than kg_embeds (e.g., mask is [B,3] but embeds are [B,60,H])
            mask_len = kg_attn.size(1)  # e.g., 3
            embed_len = kg_embeds.size(1)  # e.g., 60
            if kg_attn.dim() == 2 and embed_len % mask_len == 0:
                rep = embed_len // mask_len   # e.g., 60//3 = 20
                if rep > 1:
                    kg_attn = kg_attn.repeat_interleave(rep, dim=1)  # [B, embed_len]
            
            # Update Lkg to reflect actual embedding length after expansion
            Lkg = kg_embeds.size(1)

            kg_embeds = kg_embeds * kg_attn.unsqueeze(-1)

        # --- concat KG prefix + tokens ---
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)          # [B, Lkg+Lt, H]

        # --- attention mask & labels (types/devices consistent) ---
        if attention_mask is None:
            attention_mask = torch.ones((B, Lt), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
        new_attention_mask = torch.cat((kg_attn, attention_mask), dim=1)    # [B, Lkg+Lt]

        if labels is None:
            labels = torch.full((B, Lt), fill_value=-100, dtype=torch.long, device=device)
        else:
            labels = labels.to(device=device, dtype=torch.long)
        prefix_labels = torch.full((B, Lkg), fill_value=-100, dtype=torch.long, device=device)
        new_labels = torch.cat((prefix_labels, labels), dim=1)              # [B, Lkg+Lt]

        # after changing sequence length, let the backbone infer positions
        position_ids = None

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
        
        """
        if structure_labels is not None:
            head_ids = embedding_ids[:, 0]
            tail_ids = embedding_ids[:, 2]

            # 获取实体嵌入 (B, dim)
            head_vec = self.embeddings.ent_embeddings(head_ids)
            tail_vec = self.embeddings.ent_embeddings(tail_ids)

            # 简单打分函数：内积
            score = (head_vec * tail_vec).sum(dim=1)  # (B,)
            repurposing_loss = nn.functional.binary_cross_entropy_with_logits(score, structure_labels.float())

            # 总损失 = 语言模型损失 + repurposing 二分类损失
            output.loss = output.loss + 0.5 * repurposing_loss

        return output
        """

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
        
        # 使用普通 Embedding 而不是 from_pretrained，确保能正确移动到 GPU
        num_ent = pretrain_ent_embs.shape[0]
        num_rel = pretrain_rel_embs.shape[0]
        self.pretrain_dim = pretrain_ent_embs.shape[1]
        
        self.ent_embeddings = nn.Embedding(num_ent, self.pretrain_dim)
        self.rel_embeddings = nn.Embedding(num_rel, self.pretrain_dim)
        
        # 手动复制权重
        with torch.no_grad():
            self.ent_embeddings.weight.copy_(pretrain_ent_embs.float())
            self.rel_embeddings.weight.copy_(pretrain_rel_embs.float())
        
        # 冻结 pretrain embeddings
        self.ent_embeddings.weight.requires_grad = False
        self.rel_embeddings.weight.requires_grad = False
        
        self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)
    

    def forward(self, triple_ids):
        # main training stage
        if triple_ids.shape[1] == 3:
            head, relation, tail = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2]
            h = self.ent_embeddings(head)
            r = self.rel_embeddings(relation)
            t = self.ent_embeddings(tail)
            pretrain_embs = torch.stack((h, r, t), dim=1)
            if torch.is_complex(pretrain_embs):
                pretrain_embs = pretrain_embs.real
            pretrain_embs = pretrain_embs.to(dtype=self.adapter.weight.dtype)
            prefix = self.adapter(pretrain_embs).reshape(-1, 3*self.num_prefix, self.llm_dim)
            return prefix
        # entity-aware pre-funing
        else:
            ent = triple_ids.reshape(-1,)
            emb = self.ent_embeddings(ent)
            if torch.is_complex(emb):
                emb = emb.real
            emb = emb.to(dtype=self.adapter.weight.dtype)
            prefix = self.adapter(emb).reshape(-1, self.num_prefix, self.llm_dim)
            # print(prefix.shape)
            return prefix

