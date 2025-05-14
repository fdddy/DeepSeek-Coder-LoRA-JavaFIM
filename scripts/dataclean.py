import os
import re
import hashlib
from pathlib import Path

def remove_comments(code):
    """
    去除Java代码中的单行注释、块注释和Javadoc注释。
    """
    # 去除单行注释
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    
    # 去除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 去除Javadoc注释
    code = re.sub(r'/\*\*.*?\*/', '', code, flags=re.DOTALL)
    
    return code

def is_only_comments(code):
    """
    判断代码是否只包含注释或为空。
    """
    cleaned_code = remove_comments(code)
    return not cleaned_code.strip()

def hash_code(content):
    """
    生成代码内容的哈希值（SHA-256）。
    用于去重相似或相同的代码。
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def clean_code_file(input_file, output_file, seen_hashes):
    """
    清洗单个Java文件，去除注释并去除空白或无效代码，
    同时进行去重，确保相同代码不被多次处理。
    """
    if not os.path.isfile(input_file):  # 检查文件是否为普通文件
        return False

    with open(input_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # 去除注释
    cleaned_code = remove_comments(code)

    # 如果文件仅包含注释或空白，则跳过
    if not cleaned_code.strip():
        return False

    # 计算清洗后的代码的哈希值
    code_hash = hash_code(cleaned_code)
    
    # 如果该哈希值已经出现过，说明是重复的，跳过
    if code_hash in seen_hashes:
        return False

    # 将哈希值记录下来，避免重复
    seen_hashes.add(code_hash)

    # 将清洗后的代码写入新文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(cleaned_code)
    
    return True

def clean_all_files(input_dir, output_dir):
    """
    清洗指定目录下的所有Java文件并保存至新目录，同时去重。
    支持递归遍历子文件夹。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cleaned_count = 0
    seen_hashes = set()  # 用于记录处理过的代码哈希值

    # 使用Path.rglob()递归查找所有的java文件
    java_files = Path(input_dir).rglob("*.java")

    for file_path in java_files:
        output_file_path = os.path.join(output_dir, file_path.relative_to(input_dir))

        # 创建文件夹结构
        if not os.path.exists(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))

        if clean_code_file(file_path, output_file_path, seen_hashes):
            cleaned_count += 1
            print(f"已清洗文件: {file_path}")
        else:
            print(f"跳过文件（空白、仅注释、重复或目录）: {file_path}")

    print(f"共清洗 {cleaned_count} 个有效文件，去重后剩余有效文件。")

if __name__ == "__main__":
    # 输入目录：原始的 Java 代码目录
    # input_dir = '/root/autodl-tmp/raw_data'
    input_dir = '../data/raw_data'
    
    # 输出目录：清洗后的 Java 代码保存目录
    # output_dir = '/root/autodl-tmp/cleaned_data'
    output_dir = '../data/cleaned_data'

    # 清洗目录下所有 Java 文件并进行去重
    clean_all_files(input_dir, output_dir)
