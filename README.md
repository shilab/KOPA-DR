# KoPA-DR: DDI-aware Safety Regularization for Biomedical Knowledge Graph Completion

Official implementation of **KoPA-DR**, a biomedical extension of the KoPA framework for structure-aware large language model reasoning over biomedical knowledge graphs.

KoPA-DR introduces **DDI-aware topology tracing** and **safety-aware regularization** to improve the biological plausibility and clinical safety of biomedical KG completion and drug repurposing.

---

# Overview

Large Language Models (LLMs) have recently shown promising capabilities for knowledge graph completion (KGC). Existing approaches such as KoPA inject graph structural information into frozen LLMs through soft-prefix adaptation.

However, generic KGC frameworks do not explicitly account for biomedical safety constraints. In biomedical applications, incorrect drug interaction reasoning may lead to clinically unsafe predictions.

To address this limitation, **KoPA-DR** extends KoPA with:

* DDI-aware topology tracing
* Safety-aware regularization
* Biomedical graph adaptation on PrimeKG
* Drug repurposing evaluation

---

# Key Features

* Structure-aware LLM reasoning using graph prefix adaptation
* DDI-aware safety regularization
* Disease-centric drug interaction topology tracing
* Parameter-efficient LoRA adaptation
* Biomedical KG completion on PrimeKG
* Drug repurposing evaluation pipeline

---

# Method Overview

KoPA-DR consists of two stages:

## 1. KoPA Backbone (Inherited)

The original KoPA framework injects graph structural information into a frozen LLM through prefix-based adaptation.

Pipeline:

```text
Biomedical KG
→ Graph Structure Encoding
→ Prefix-based Structure Injection
→ Frozen LLM Reasoning
→ KG Completion
```

## 2. DDI-aware Safety Learning (Our Contribution)

KoPA-DR introduces disease-centric DDI topology tracing and safety-aware regularization.

The DDI topology captures:

* co-treatment overlap
* adverse drug reaction (ADR) overlap
* local interaction topology proximity

These safety-aware signals regularize prefix optimization toward clinically safer biomedical reasoning.

---

# Installation

```bash
git clone https://github.com/Yang-Zhao-CIS-TU/KOPA-DR.git

cd KOPA-DR

pip install -r requirements.txt
```

---

# Training

Example DDI-aware training:

```bash
python finetune_kopa_ddi.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --data_path data/sample/primekg_train_sub50k_posneg_named_fixed.jsonl \
  --ddi_lambda 0.1
```

---

# Evaluation

```bash
python evaluate_kopa_ddi.py
```



---

# Acknowledgement

This repository builds upon the original KoPA framework:

[https://github.com/zjukg/KoPA](https://github.com/zjukg/KoPA)

We thank the original KoPA authors for releasing their codebase.

---

# License

This project follows the same license as the original KoPA repository.

