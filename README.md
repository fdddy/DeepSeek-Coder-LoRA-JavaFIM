## 📌 注意事项

- 基于`LLaMA-Factory` 执行微调，微调数据集已经集成 `Deepseekcoder-fim-lora-llamafy/data`目录下。
- 处理和输出数据保存在 `data/` 和 `output/` 目录下。**文件中代码部分的文件路径使用绝对路径（如模型位置等），因此如需运行需要进一步修改文件路径。**
- 请根据实际模型路径和配置修改 `train_fim.sh` 和脚本中的超参数。

## 📁 项目结构

```bash
.
├── data				    # 存放处理后的数据集
│   ├── cleaned_data	
│   ├── eval_dataset.json	# 用于评估的数据集
│   ├── mixed_finetune_dataset.json # 微调数据集
│   └── raw_data
├── output					# 模型训练输出及中间结果
├── requirements.txt
├── scripts					# 核心脚本目录
│   ├── dataclean.py              # 清洗原始Java代码
│   ├── datamerge.py              # 合并多份数据为训练集
│   ├── dataset.py                # 加载与处理训练数据
│   ├── evaluate_fim_compare.py   # 评估不同FIM补全方式效果
│   ├── java_processing.log       # 数据处理日志文件
│   ├── prepare_fim_data.py       # 生成FIM格式的训练数据
└── test.py                   # 加载模型并进行FIM补全测试
└── train_fim.sh              # LoRA微调训练脚本
```

## 🚀 使用说明

### 1. 数据预处理

依次执行以下脚本：

```bash
python scripts/dataset.py
python scripts/dataclean.py
python scripts/prepare_fim_data.py
python scripts/datamerge.py
```

### 2. 模型训练

执行训练脚本开始微调：

```bash
bash train_fim.sh
```

### 3. 评估与测试

评估FIM补全效果：

```bash
python scripts/evaluate_fim_compare.py
```

测试推理效果：

```bash
python test.py
```