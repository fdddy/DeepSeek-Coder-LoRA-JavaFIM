import json
from datasets import load_dataset
import random
from tqdm import tqdm

def convert_bigcode_to_alpaca(example):
    """转换BigCode特殊格式到Alpaca格式"""
    return {
        "instruction": example.get("instruction", ""),
        "input": example.get("prompt", ""),
        "output": example.get("response", "")
    }

def convert_general_to_alpaca(example):
    """转换通用Alpaca格式数据集"""
    return {
        "instruction": example.get("instruction", ""),
        "input": example.get("input", ""),
        "output": example.get("output", "")
    }

def load_and_process_dataset(config):
    """加载并处理单个数据集"""
    try:
        # 加载数据集
        if config['name'] == "json":
            dataset = load_dataset(config['name'], data_files=config['path'])['train']
        else:
            dataset = load_dataset(config['name'])['train']
        
        # 随机采样
        if config.get('max_samples'):
            if len(dataset) > config['max_samples']:
                indices = random.sample(range(len(dataset)), config['max_samples'])
                dataset = dataset.select(indices)
            else:
                print(f"{config['name']} 数据集样本不足，使用全部 {len(dataset)} 条数据")

        # 转换格式
        converter = convert_bigcode_to_alpaca if config['type'] == 'bigcode' else convert_general_to_alpaca
        converted_data = []
        
        for example in tqdm(dataset, desc=f"处理 {config['name']}"):
            try:
                converted = converter(example)
                # 过滤无效数据
                if len(converted["output"].strip()) > 10:  # 至少10个字符
                    converted_data.append(converted)
            except Exception as e:
                print(f"转换失败: {e}\n原始数据: {example}")
        
        return converted_data
    
    except Exception as e:
        print(f"加载数据集 {config['name']} 失败: {str(e)}")
        return []

# 数据集配置
DATASET_CONFIGS = [
    {
        "name": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "type": "bigcode",
        "max_samples": 19000,
        "desc": "BigCode自对齐数据集（需特殊转换）"
    },
    {
        "name": "json",
        "path": "/root/LLaMA-Factory/data/alpaca_fim_dataset.json",
        "type": "fim",
        "max_samples": 30000,
        "desc": "代码填充数据集"
    },
    {
        "name": "json",
        "path": "/root/LLaMA-Factory/data/alpaca_zh_demo.json",
        "type": "zh",
        "max_samples": 1000,
        "desc": "中文指令数据集"
    }
]

if __name__ == "__main__":
    # 处理所有数据集
    all_data = []
    for cfg in DATASET_CONFIGS:
        print(f"\n正在处理: {cfg['desc']}")
        data = load_and_process_dataset(cfg)
        all_data.extend(data)
        print(f"已转换 {len(data)} 条有效数据")

    # 打乱并分割数据集
    random.shuffle(all_data)
    total_samples = len(all_data)
    print(f"\n总有效样本数: {total_samples}")
    
    # 确保评估集不包含训练数据
    eval_size = min(200, int(total_samples * 0.05))  # 5%或200条
    eval_data = all_data[:eval_size]
    train_data = all_data[eval_size:]
    
    # 保存数据集
    def save_json(data, path):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"成功保存 {len(data)} 条数据到 {path}")
        except Exception as e:
            print(f"保存失败 {path}: {str(e)}")

    # train_path = "/root/LLaMA-Factory/data/mixed_finetune_dataset.json"
    # eval_path = "/root/deepseekcode-lora/data/eval_dataset.json"

    train_path = "../data/mixed_finetune_dataset.json"
    eval_path = "../data/eval_dataset.json"

    save_json(train_data, train_path)
    save_json(eval_data, eval_path)

    # 打印最终统计
    print(f"""
        ========== 最终数据集统计 ==========
        原始数据分布:
        - BigCode: {len([d for d in all_data if d.get('concepts')])} 条
        - FIM:     {len([d for d in all_data if 'fim_middle' in d.get('input','')])} 条
        - 中文:     {len([d for d in all_data if any(c >= chr(0x4e00) for c in d.get('output',''))])} 条
        
        最终分割:
        - 训练集: {len(train_data)} 条
        - 评估集: {len(eval_data)} 条
        ====================================
        """)