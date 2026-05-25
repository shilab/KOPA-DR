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

