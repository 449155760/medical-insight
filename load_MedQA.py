import os
import glob
import json
# 适配新版 LangChain 的导入方式
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# === 路径配置核心修改 ===
# 1. 获取当前脚本所在的绝对路径 (即 pythonProject3 文件夹路径)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 修改文件夹名称：根据你的截图，文件夹叫 "MedQA"，不是 "data_clean"
DATA_FOLDER_NAME = "MedQA"

# 3. 拼接基础路径
BASE_DIR = os.path.join(CURRENT_DIR, DATA_FOLDER_NAME)

# 4. 拼接具体子路径 (根据 Image 1 的结构)
# 教科书路径: MedQA/textbooks/zh_paragraph
TEXTBOOK_DIR = os.path.join(BASE_DIR, "textbooks", "zh_paragraph")

# 问题路径: MedQA/questions/Mainland/4_options/test.jsonl
# 注意：请确保你的 Mainland 文件夹下确实有 "4_options" 这个子文件夹
QUESTION_FILE = os.path.join(BASE_DIR, "questions", "Mainland", "4_options", "test.jsonl")

# === 调试打印 ===
print(f"DEBUG: 脚本所在位置 -> {CURRENT_DIR}")
print(f"DEBUG: 尝试读取数据根目录 -> {BASE_DIR}")
print(f"DEBUG: 尝试读取教科书目录 -> {TEXTBOOK_DIR}")
print(f"DEBUG: 尝试读取问题文件 -> {QUESTION_FILE}")


def load_and_chunk_corpus(limit_files=None):
    """
    加载教科书并切分。
    """
    print(f"\n--- [1/2] 正在加载教科书数据 ---")

    if not os.path.exists(TEXTBOOK_DIR):
        print(f"❌ 严重错误: 找不到目录 {TEXTBOOK_DIR}")
        print("请检查：\n1. MedQA 文件夹下是否有 textbooks 文件夹？\n2. textbooks 下是否有 zh_paragraph？")
        return []

    txt_files = glob.glob(os.path.join(TEXTBOOK_DIR, "*.txt"))

    if not txt_files:
        print(f"⚠️ 警告: 目录存在，但未找到 .txt 文件。")
        return []

    if limit_files:
        txt_files = txt_files[:limit_files]
        print(f"ℹ️ 测试模式: 仅加载前 {limit_files} 个文件。")

    all_docs = []
    for file_path in txt_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"跳过文件 {os.path.basename(file_path)}: {e}")

    # 切分策略
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？"]
    )

    chunked_docs = text_splitter.split_documents(all_docs)
    print(f"✅ 教科书加载完成: {len(txt_files)} 个文件 -> {len(chunked_docs)} 个片段 (Chunks)。")
    return chunked_docs


def load_test_questions(limit=5):
    """
    加载测试集问题。
    """
    print(f"\n--- [2/2] 正在加载测试题目 ---")

    if not os.path.exists(QUESTION_FILE):
        print(f"❌ 严重错误: 找不到文件 {QUESTION_FILE}")
        # 针对 Image 1 结构的容错提示
        print(
            "提示: 请检查 MedQA/questions/Mainland 下是否有 '4_options' 文件夹？如果没有，可能路径是 questions/Mainland/test.jsonl")
        return []

    questions = []
    with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                q_text = data['question']
                options_str = "\n".join([f"{k}: {v}" for k, v in data['options'].items()])
                full_query = f"{q_text}\n选项:\n{options_str}"

                questions.append({
                    "query": full_query,
                    "ground_truth": data['answer_idx'],
                    "ground_truth_text": data['options'][data['answer_idx']],
                    "raw_data": data
                })
            except Exception as e:
                continue

    print(f"✅ 测试题加载完成: 准备测试 {len(questions)} 个问题。")
    return questions


# 只有直接运行此文件时才执行测试
if __name__ == "__main__":
    # 测试加载1个文件和1个问题
    load_and_chunk_corpus(limit_files=1)
    load_test_questions(limit=1)