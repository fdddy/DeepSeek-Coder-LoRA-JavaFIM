from datasets import load_dataset
from pathlib import Path
import os
import re

def sanitize_path(path):
    """清理非法文件路径字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', path)

def save_java_file(content, repo_name, file_path, output_dir="java_files"):
    """保存Java文件到本地"""
    # 构建完整保存路径
    safe_repo = sanitize_path(repo_name)
    safe_path = sanitize_path(file_path)
    full_path = Path(output_dir) / safe_repo / safe_path
    
    # 确保目录存在
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入文件
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return str(full_path)

def main():
    # 流式加载数据集
    dataset = load_dataset(
        "bigcode/the-stack",
        data_dir="data/java",
        split="train",
        streaming=True
    )

    # 创建输出目录
    # output_dir = "/root/autodl-tmp/raw_data"
    output_dir = "../data/raw_data"

    os.makedirs(output_dir, exist_ok=True)

    # 计数器
    file_count = 0
    skipped_files = 0

    # 处理数据集
    for sample in dataset:
        try:
            # 应用过滤条件
            if not (
                sample['avg_line_length'] <= 100 and
                sample['max_line_length'] <= 1000 and
                sample['alphanum_fraction'] >= 0.25
            ):
                skipped_files += 1
                continue

            # 获取文件元数据
            repo_name = sample.get('max_stars_repo_name', 'unknown_repo')
            file_path = sample.get('max_stars_repo_path', 'unknown_path.java')
            
            # 保存Java文件
            save_path = save_java_file(
                content=sample['content'],
                repo_name=repo_name,
                file_path=file_path,
                output_dir=output_dir
            )
            
            file_count += 1
            if file_count % 100 == 0:
                print(f"已保存 {file_count} 个文件，最新文件：{save_path}")

        except Exception as e:
            print(f"处理文件失败：{str(e)}")
            skipped_files += 1

    print(f"\n处理完成！")
    print(f"成功保存文件数：{file_count}")
    print(f"跳过文件数：{skipped_files}")
    print(f"文件保存目录：{output_dir}")

if __name__ == "__main__":
    main()