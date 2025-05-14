import os
import json
import random
import logging
from pathlib import Path
import javalang
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename='java_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class JavaProcessor:
    def __init__(self, mask_lines=6, min_context=10, max_context=100):
        self.mask_lines = mask_lines
        self.min_context = min_context
        self.max_context = max_context
        self.instruction = "根据Java代码上下文补全<｜fim▁hole｜>标记处的代码，只需返回需要补全的代码部分，不要包含注释和已有代码。"

    def _get_valid_span(self, code: str) -> dict:
        """安全获取有效代码区间"""
        try:
            tree = javalang.parse.parse(code)
            lines = code.split('\n')
            candidates = []

            def process_node(node):
                """递归处理语法节点"""
                if isinstance(node, javalang.tree.BlockStatement):
                    start_pos = getattr(node, 'position', None)
                    if start_pos and start_pos.line:
                        start_line = start_pos.line - 1
                        end_line = self._get_end_line(node, len(lines)-1)
                        if end_line > start_line and (end_line - start_line) >= self.mask_lines + 2:
                            candidates.append((start_line, end_line))

                for child in getattr(node, 'children', []):
                    if isinstance(child, list):
                        for item in child:
                            if isinstance(item, javalang.ast.Node):
                                process_node(item)
                    elif isinstance(child, javalang.ast.Node):
                        process_node(child)

            process_node(tree)
            return random.choice(candidates) if candidates else None

        except Exception as e:
            logging.error(f"语法解析失败: {str(e)}")
            return None

    def _get_end_line(self, node, default):
        """安全获取结束行号"""
        if hasattr(node, 'end_position') and node.end_position:
            return node.end_position.line - 1
        # 回退方案：查找最后一个子节点的结束位置
        for child in reversed(list(getattr(node, 'children', []))):
            if hasattr(child, 'end_position'):
                return child.end_position.line - 1
        return default

    def process_file(self, file_path: Path):
        """处理单个Java文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            if len(code) < 100:  # 过滤过短文件
                return None

            span = self._get_valid_span(code)
            if not span:
                return None

            start, end = span
            mask_start = random.randint(start + 1, end - self.mask_lines - 1)
            mask_end = mask_start + self.mask_lines

            lines = code.split('\n')
            prefix = '\n'.join(lines[max(0, mask_start-self.max_context):mask_start])
            suffix = '\n'.join(lines[mask_end:mask_end+self.max_context])
            output = '\n'.join(lines[mask_start:mask_end])

            return {
                "instruction": self.instruction,
                "input": f"<｜fim▁begin｜>{prefix}\n<｜fim▁hole｜>\n{suffix}<｜fim▁end｜>",
                "output": output
            }
            # return {
            #     "prompt": f"<｜fim▁begin｜>{prefix}\n<｜fim▁hole｜>\n{suffix}<｜fim▁end｜>",
            #     "golden_answer": output
            # }

        except Exception as e:
            logging.error(f"处理文件失败 {file_path}: {str(e)}")
            return None


def main():
    processor = JavaProcessor()
    # input_dir = "/root/autodl-tmp/cleaned_data"
    # output_path = "/root/LLaMA-Factory/data/alpaca_fim_dataset.json"
    # output_path = "/root/deepseekcode-lora/data/eval_dataset.json"
    input_dir = "../data/cleaned_data"
    output_path = "../data/alpaca_fim_dataset.json"
    
    # 安全获取所有Java文件（排除目录）
    java_files = [f for f in Path(input_dir).rglob("*.java") if f.is_file()]
    random.shuffle(java_files)
    logging.info(f"发现 {len(java_files)} 个Java文件")

    samples = []
    processed_samples = 0  # 用于记录处理的样本数
    for file_path in tqdm(java_files, desc="Processing"):
        try:
            if sample := processor.process_file(file_path):
                if sample and 50 < len(sample["input"]) < 10000:
                # if sample and 50 < len(sample["prompt"]) < 10000:
                    samples.append(sample)
                    processed_samples += 1

            if len(samples) % 100 == 0:
                logging.info(f"已处理 {len(samples)} 个样本")

            # if processed_samples >= 50:
            #         break  
        
        except Exception as e:
            print(f"处理文件失败 {file_path}: {str(e)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logging.info(f"保存完成，共 {len(samples)} 个样本")

if __name__ == "__main__":
    main()