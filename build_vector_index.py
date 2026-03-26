import time
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ================= 配置 =================
# 1. 模型选择: BAAI/bge-small-zh-v1.5 是目前中文语义匹配的性价比之王
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 2. 数据库配置
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "Ljy449155760"  # 你的密码


# ========================================

def create_index():
    print(f"⏳ 正在加载 AI 模型: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    print("⏳ 正在读取图谱中的所有实体...")
    # 读取所有有名字的节点 (疾病、症状、药物等)
    fetch_query = """
    MATCH (n) 
    WHERE (n:Disease OR n:Symptom OR n:Drug OR n:Check) AND n.name IS NOT NULL
    RETURN elementId(n) AS id, n.name AS name
    """

    with driver.session() as session:
        results = list(session.run(fetch_query))

    print(f"📊 共找到 {len(results)} 个实体，开始计算向量 (Embedding)...")

    # 批量计算向量
    names = [r["name"] for r in results]
    ids = [r["id"] for r in results]

    # 生成 512 维度的向量
    embeddings = model.encode(names, normalize_embeddings=True)

    print("⏳ 正在将向量写回 Neo4j...")

    # 批量更新数据库
    update_query = """
    UNWIND $batch AS row
    MATCH (n) WHERE elementId(n) = row.id
    SET n.embedding = row.vector
    """

    batch_size = 1000
    total = len(ids)

    with driver.session() as session:
        for i in range(0, total, batch_size):
            batch_data = []
            for j in range(i, min(i + batch_size, total)):
                batch_data.append({
                    "id": ids[j],
                    "vector": embeddings[j].tolist()
                })
            session.run(update_query, batch=batch_data)
            print(f"   已处理 {min(i + batch_size, total)}/{total}...")

    print("⏳ 正在创建向量索引 (Vector Index)...")

    # 创建索引以便快速检索
    # 注意: 512 是 bge-small 的维度，如果你换模型，这里要改
    create_idx_query = """
    CREATE VECTOR INDEX medical_entity_index IF NOT EXISTS
    FOR (n:Disease|Symptom|Drug|Check)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 512,
      `vector.similarity_function`: 'cosine'
    }}
    """
    try:
        with driver.session() as session:
            session.run(create_idx_query)
    except Exception as e:
        print(f"索引创建提示 (如果已存在可忽略): {e}")

    driver.close()
    print("🎉 恭喜！图谱大脑升级完毕，现在支持语义搜索了！")


if __name__ == "__main__":
    create_index()