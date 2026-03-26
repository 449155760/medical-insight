import pandas as pd
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import jieba

# 引入您的模块 (确保这俩文件在当前目录)
try:
    from reasoning_module import ReasoningModule
    from factual_module import FactualModule
except ImportError:
    print("❌ 错误：未找到 reasoning_module.py 或 factual_module.py")
    print("请确保它们在当前目录下！")
    sys.exit(1)

# ================= 配置区域 (基于您的 Ablation.py) =================
CONFIG = {
    # 路径配置
    "chroma_path":"/home/lab207/data/ljy/data/chroma_db_medqa_chunked",
    "table_path": "/home/lab207/data/ljy/data/medical_data.xlsx",

    # 输入文件：使用您上传的 Ablation 结果文件作为问题源
    "input_file": "System_Ablation_Results_20260130_014053.xlsx",  # 如果文件名不同请修改
    "output_file": "True_Ours_NoAgent_Relevance.xlsx",

    # 数据库配置
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "Ljy449155760",

    # API 配置
    "api_key": "ollama",
    "base_url": "http://localhost:11434/v1",
    "model_name": "qwen2.5:7b",

    # 模型配置
    "embedding_model": "/home/lab207/data/ljy/Baselines/models/bge-small-zh-v1.5",
    "rerank_model": "/home/lab207/data/ljy/Baselines/models/bge-reranker-base",
}


# =======================================================

def get_relevance_score(client, question, context):
    """LLM 裁判：判断上下文是否包含答案"""
    if not context: return 0

    prompt = f"""你是一个严格的评审员。请评估【参考资料】是否包含回答【问题】所需的关键信息。

【问题】：{question}
【参考资料】：{str(context)[:1500]}

请仅输出 JSON 格式结果：{{"score": 1}} (若包含有用信息) 或 {{"score": 0}} (若完全无关)。
不要输出任何其他解释。
"""
    try:
        resp = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        return data.get("score", 0)
    except:
        return 0


def main():
    print("🚀 正在初始化系统 (No_Agent 模式)...")

    # 1. 连接数据库
    print("   ├── 连接 Neo4j...")
    driver = GraphDatabase.driver(CONFIG["neo4j_uri"], auth=(CONFIG["neo4j_user"], CONFIG["neo4j_password"]))

    print("   ├── 连接 ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CONFIG["chroma_path"])
    collection = chroma_client.get_collection("medqa_corpus")

    # 2. 加载模型
    print("   ├── 加载 Embedding 模型...")
    encoder = SentenceTransformer(CONFIG["embedding_model"])
    print("   ├── 加载 Rerank 模型...")
    reranker = CrossEncoder(CONFIG["rerank_model"])

    # 3. 初始化模块
    reasoning_mod = ReasoningModule(CONFIG, driver, encoder_model=encoder)

    # 需要先加载文档构建 BM25 (这可能需要一点时间)
    print("   ├── 构建 BM25 索引 (读取 Chroma)...")
    all_docs = collection.get()['documents']
    factual_mod = FactualModule(CONFIG, collection, all_docs, encoder=encoder, reranker=reranker)

    # 4. 读取问题
    print(f"📂 读取问题文件: {CONFIG['input_file']} ...")
    if CONFIG['input_file'].endswith('.csv'):
        df = pd.read_csv(CONFIG['input_file'])
    else:
        df = pd.read_excel(CONFIG['input_file'])

    print(f"📊 待评测数据量: {len(df)} 条")

    client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])
    results = []

    # 5. 开始循环
    print("🚀 开始重现检索与评分...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        query = row.get('query', row.get('question'))

        # === 核心：重现 No_Agent 检索逻辑 ===
        # 1. 图谱路
        graph_ctx = reasoning_mod.run_tog(query)
        # 2. 文本路
        text_ctx = factual_module_retrieve_wrapper(factual_mod, query)  # 辅助函数

        # 3. 拼接上下文
        full_context = ""
        if graph_ctx:
            full_context += f"【图谱路径】:\n{graph_context}\n\n"
        if text_ctx:
            full_context += f"【参考资料】:\n{text_ctx}"

        # 4. LLM 打分
        score = get_relevance_score(client, query, full_context)

        results.append({
            "id": row.get('id', idx),
            "query": query,
            "retrieved_context": full_context,
            "relevance": score
        })

    # 6. 保存
    res_df = pd.DataFrame(results)
    avg_rel = res_df['relevance'].mean()

    print("\n" + "=" * 40)
    print(f"✅ 计算完成！")
    print(f"🔥 True No_Agent Relevance: {avg_rel:.4f}")
    print("=" * 40)

    res_df.to_excel(CONFIG['output_file'], index=False)
    print(f"💾 结果已保存至: {CONFIG['output_file']}")


# 辅助：因为 FactualModule.retrieve 返回的是 list，需要转 string
def factual_module_retrieve_wrapper(mod, query):
    candidates = mod.retrieve(query)  # 返回 list of strings
    if not candidates: return ""
    return "\n".join(candidates[:5])  # 取 Top 5


if __name__ == "__main__":
    main()