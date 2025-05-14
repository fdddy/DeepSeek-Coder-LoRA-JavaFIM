import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

def load_model(base_model_path, lora_path=None):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if lora_path:
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    
    return tokenizer, model

def generate_completion(tokenizer, model, prompt, max_new_tokens=128):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    completion_start = prompt.find("<|fim_middle|>") + len("<|fim_middle|>")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[completion_start:].strip()

def evaluate_models(base_model_path, lora_model_path, eval_data_path, output_base_path, output_lora_path):
    tokenizer, base_model = load_model(base_model_path)
    _, lora_model = load_model(base_model_path, lora_model_path)

    results_base = []
    results_lora = []

    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Evaluating"):
            item = json.loads(line)
            prompt = item["input"]
            
            if "<|fim_middle|>" not in prompt:
                continue
                
            # Base model prediction
            pred_base = generate_completion(tokenizer, base_model, prompt)
            results_base.append({
                "instruction": item["instruction"],
                "input": prompt,
                "output": pred_base,
                "golden": item["output"]
            })
            
            # LoRA model prediction
            pred_lora = generate_completion(tokenizer, lora_model, prompt)
            results_lora.append({
                "instruction": item["instruction"],
                "input": prompt,
                "output": pred_lora,
                "golden": item["output"]
            })

    # Save results
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)
    with open(output_base_path, 'w', encoding='utf-8') as f:
        json.dump(results_base, f, indent=2)

    os.makedirs(os.path.dirname(output_lora_path), exist_ok=True)
    with open(output_lora_path, 'w', encoding='utf-8') as f:
        json.dump(results_lora, f, indent=2)

class EMScorer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.fim_token = "<|fim_middle|>"

    def _clean_code(self, code: str) -> str:
        """清理代码：移除注释和空行"""
        lines = []
        for line in code.split('\n'):
            line = line.split('//')[0].strip()
            if line:
                lines.append(line)
        return '\n'.join(lines[:6])  # 只比较前6行

    def evaluate_sample(self, item: dict) -> float:
        prompt = item["input"]
        golden = self._clean_code(item["golden"])
        
        try:
            pred = generate_completion(self.tokenizer, self.model, prompt)
            pred = self._clean_code(pred)
            return float(pred == golden)
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return 0.0

def run_comparison(eval_data_path, output_base_path, output_lora_path):
    # 加载结果
    with open(output_base_path, 'r') as f:
        base_results = json.load(f)
    with open(output_lora_path, 'r') as f:
        lora_results = json.load(f)
    
    # 初始化评分器
    scorer = EMScorer("/root/autodl-tmp/deepseek-coder-1.3b-base")
    
    base_scores = []
    lora_scores = []
    
    for base_item, lora_item in zip(base_results, lora_results):
        base_scores.append(scorer.evaluate_sample(base_item))
        lora_scores.append(scorer.evaluate_sample(lora_item))
    
    print(f"Base Model EM Score: {np.mean(base_scores)*100:.2f}%")
    print(f"LoRA Model EM Score: {np.mean(lora_scores)*100:.2f}%")

if __name__ == "__main__":
    base_model = "/root/autodl-tmp/deepseek-coder-1.3b-base"
    # lora_model = "/root/LLaMA-Factory/lora_deepseekcode_fim/saves/deepseek-coder-1.3b/lora/fim_full"
    lora_model = "/root/LLaMA-Factory/deepseekcode-lora/output/lora_saves"
    eval_data = "../data/eval_dataset.json"
    
    # 生成预测结果
    evaluate_models(
        base_model,
        lora_model,
        eval_data,
        "output/base_preds.json",
        "output/lora_preds.json"
    )
    
    # 执行对比评估
    run_comparison(
        eval_data,
        "output/base_preds.json",
        "output/lora_preds.json"
    )