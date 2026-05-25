import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import LlamaForCausalLM, LlamaTokenizer

# ========== Prompt 模板 ==========
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Please determine the correctness of the triple and response True or False.

### Input:
{}

### Response:
"""

def load_test_dataset(path):
    """加载测试数据集（支持 jsonl 和 json 格式）"""
    test_dataset = []
    if path.endswith('.jsonl'):
        with open(path, 'r') as f:
            for line in f:
                test_dataset.append(json.loads(line.strip()))
    else:
        test_dataset = json.load(open(path, "r"))
    return test_dataset


def compute_classification_metrics(answers, predictions):
    """计算分类指标：Accuracy, Precision, Recall, F1"""
    acc = accuracy_score(y_true=answers, y_pred=predictions)
    p = precision_score(y_true=answers, y_pred=predictions, zero_division=0)
    r = recall_score(y_true=answers, y_pred=predictions, zero_division=0)
    f1 = f1_score(y_true=answers, y_pred=predictions, zero_division=0)
    return acc, p, r, f1


def compute_ranking_metrics(scores, labels):
    """
    计算排名指标：MR, MRR, Hits@1, Hits@3, Hits@10
    """
    if len(scores) == 0:
        return {'MR': 0, 'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@10': 0}
    
    sorted_indices = np.argsort(-np.array(scores))
    sorted_labels = np.array(labels)[sorted_indices]
    
    positive_ranks = np.where(sorted_labels == 1)[0] + 1
    
    if len(positive_ranks) == 0:
        return {'MR': float('inf'), 'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@10': 0}
    
    mr = np.mean(positive_ranks)
    mrr = np.mean(1.0 / positive_ranks)
    hits_1 = np.mean(positive_ranks <= 1)
    hits_3 = np.mean(positive_ranks <= 3)
    hits_10 = np.mean(positive_ranks <= 10)
    
    return {
        'MR': mr,
        'MRR': mrr,
        'Hits@1': hits_1,
        'Hits@3': hits_3,
        'Hits@10': hits_10
    }


def get_model_confidence(model, tokenizer, input_embeds, device):
    """获取模型对 True/False 的置信度分数"""
    with torch.no_grad():
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :]
        
        true_token_id = tokenizer.encode("True", add_special_tokens=False)[0]
        false_token_id = tokenizer.encode("False", add_special_tokens=False)[0]
        
        probs = torch.softmax(logits, dim=-1)
        true_prob = probs[0, true_token_id].item()
        false_prob = probs[0, false_token_id].item()
        
        total = true_prob + false_prob
        if total > 0:
            confidence = true_prob / total
        else:
            confidence = 0.5
            
    return confidence


def main():
    parser = argparse.ArgumentParser(description='KoPA/KoPA-DDI 模型评估')
    parser.add_argument('--base_model', type=str, default='./models/Llama-2-7b-hf',
                        help='LLaMA 基础模型路径')
    parser.add_argument('--lora_weights', type=str, default='./runs/sub50k_ddi',
                        help='训练输出目录（包含 adapter_model.bin 和 embeddings.pth）')
    parser.add_argument('--test_data', type=str, default='./primekg_test_subset_posneg_5x_filtered.jsonl',
                        help='测试数据路径')
    parser.add_argument('--cuda_device', type=str, default='cuda:0',
                        help='CUDA 设备')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大测试样本数（用于快速测试）')
    args = parser.parse_args()

    print("=" * 60)
    print("KoPA 模型评估")
    print("=" * 60)
    print(f"基础模型: {args.base_model}")
    print(f"LoRA 权重: {args.lora_weights}")
    print(f"测试数据: {args.test_data}")
    print(f"设备: {args.cuda_device}")
    
    # 加载测试数据
    print(f"\n加载测试数据: {args.test_data}")
    test_dataset = load_test_dataset(args.test_data)
    if args.max_samples:
        test_dataset = test_dataset[:args.max_samples]
    print(f"测试样本数: {len(test_dataset)}")
    
    # 加载 KG embeddings
    embedding_path = os.path.join(args.lora_weights, "embeddings.pth")
    print(f"\n加载 KG embeddings: {embedding_path}")
    
    if not os.path.exists(embedding_path):
        print(f"[警告] embeddings.pth 不存在，尝试从 checkpoint 提取...")
        # 尝试从 checkpoint 提取
        checkpoint_dir = None
        for item in os.listdir(args.lora_weights):
            if item.startswith("checkpoint-"):
                checkpoint_dir = os.path.join(args.lora_weights, item)
                break
        
        if checkpoint_dir:
            print(f"[信息] 找到 checkpoint: {checkpoint_dir}")
            print("[信息] 请先运行提取脚本生成 embeddings.pth")
        raise FileNotFoundError(f"找不到 {embedding_path}")
    
    kg_embeddings = torch.load(embedding_path, map_location="cpu")
    kg_embeddings = kg_embeddings.to(args.cuda_device)
    
    # 修复复数类型问题
    for name, param in kg_embeddings.named_parameters():
        if param.is_complex():
            param.data = param.data.real.float()
    for name, buf in kg_embeddings.named_buffers():
        if buf.is_complex():
            kg_embeddings._buffers[name] = buf.real.float()
    if hasattr(kg_embeddings, 'ent_embeddings'):
        if kg_embeddings.ent_embeddings.weight.is_complex():
            kg_embeddings.ent_embeddings.weight.data = kg_embeddings.ent_embeddings.weight.data.real.float()
    if hasattr(kg_embeddings, 'rel_embeddings'):
        if kg_embeddings.rel_embeddings.weight.is_complex():
            kg_embeddings.rel_embeddings.weight.data = kg_embeddings.rel_embeddings.weight.data.real.float()
    
    kg_embeddings.eval()
    
    # 加载 tokenizer
    print(f"\n加载 tokenizer: {args.base_model}")
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    
    # 加载基础模型
    print(f"\n加载基础模型: {args.base_model}")
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True
    ).to(args.cuda_device)
    
    # 加载 LoRA 权重
    print(f"\n加载 LoRA 权重: {args.lora_weights}")
    model = PeftModel.from_pretrained(
        model,
        args.lora_weights,
        torch_dtype=torch.float16,
    ).to(args.cuda_device)
    
    # 配置 token IDs
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    model.eval()
    
    # 评估
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)
    
    results = []
    answers = []
    predictions = []
    confidence_scores = []
    
    for data in tqdm(test_dataset, desc="评估进度"):
        ent = data["input"]
        ans = data["output"]
        ids = data["embedding_ids"]
        
        # 准备 KG embedding
        ids = torch.LongTensor(ids).reshape(1, -1).to(args.cuda_device)
        with torch.no_grad():
            prefix = kg_embeddings(ids)
        
        # 准备输入
        prompt = PROMPT_TEMPLATE.format(ent)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(args.cuda_device)
        
        # 获取 token embeddings 并拼接 KG prefix
        with torch.no_grad():
            token_embeds = model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((prefix, token_embeds), dim=1)
        
        # 获取置信度分数（用于排名指标）
        confidence = get_model_confidence(model, tokenizer, input_embeds, args.cuda_device)
        confidence_scores.append(confidence)
        
        # 生成回复
        with torch.no_grad():
            generate_ids = model.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码回复
        context = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.replace(context, "").strip()
        
        # 解析答案和预测
        if "True" in ans:
            answers.append(1)
        else:
            answers.append(0)
            
        if "True" in response:
            predictions.append(1)
        else:
            predictions.append(0)
        
        results.append({
            "input": ent,
            "answer": ans,
            "predict": response,
            "confidence": confidence
        })
    
    # 计算分类指标
    acc, precision, recall, f1 = compute_classification_metrics(answers, predictions)
    
    # 计算排名指标
    ranking_metrics = compute_ranking_metrics(confidence_scores, answers)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    print("\n【分类指标】")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\n【排名指标】")
    print(f"  MR:        {ranking_metrics['MR']:.4f}")
    print(f"  MRR:       {ranking_metrics['MRR']:.4f}")
    print(f"  Hits@1:    {ranking_metrics['Hits@1']:.4f}")
    print(f"  Hits@3:    {ranking_metrics['Hits@3']:.4f}")
    print(f"  Hits@10:   {ranking_metrics['Hits@10']:.4f}")
    
    # 保存详细结果
    output_path = os.path.join(args.lora_weights, "eval_results.json")
    eval_summary = {
        "classification": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "ranking": ranking_metrics,
        "num_samples": len(test_dataset),
        "predictions": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
