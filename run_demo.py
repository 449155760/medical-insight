import os
import chromadb
from neo4j import GraphDatabase
from tqdm import tqdm
from main_system import MedicalQASystem

# 1. 获取当前脚本所在的根目录 (D:\pycharm\pythonProject3)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 定义数据文件夹路径 (D:\pycharm\pythonProject3\data)
DATA_DIR = os.path.join(BASE_DIR, "data")

CONFIG = {
    "api_key": "sk-18cdce104dc64150a6039c86cef0fd4f",
    "base_url": "https://api.deepseek.com",

    "embedding_model": "BAAI/bge-small-zh-v1.5",

    # 向量库通常在根目录 (如果你没动过它的话)
    "chroma_path": os.path.join(BASE_DIR, "chroma_db_medqa_chunked"),

    # 【👇 核心修改 👇】指向 data 子文件夹
    "table_path": os.path.join(DATA_DIR, "medical_data.xlsx"),
    "synonym_path": os.path.join(DATA_DIR, "synonyms.txt"),

    # Neo4j 配置
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "Ljy449155760"  # 【记得改密码】
}


# ================= 优化后的加载函数 =================

def load_docs_from_chroma(chroma_path, collection_name="medqa_corpus"):
    """
    🚀 极速模式：直接从 ChromaDB 加载切片数据
    不再读取原始 TXT 文件，大大减少 I/O 和分词压力
    """
    print(f"🔄 正在连接向量库: {chroma_path} ...")
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=collection_name)

        print("📥 正在从数据库拉取文档切片 (这比读TXT快得多)...")
        # 获取所有文档内容
        data = collection.get()
        documents = data['documents']

        if not documents:
            print("❌ 警告：向量库是空的！请先运行 Hybrid_RAG 构建索引。")
            return [], None

        print(f"✅ 成功加载 {len(documents)} 个文档切片用于 BM25")
        return documents, collection

    except Exception as e:
        print(f"❌ 读取向量库失败: {e}")
        return [], None


# ================= 主程序入口 =================

def main():
    print("🚀 正在初始化医疗问答系统 (优化版)...")

    # 1. 连接 Neo4j
    try:
        driver = GraphDatabase.driver(
            CONFIG["neo4j_uri"],
            auth=(CONFIG["neo4j_user"], CONFIG["neo4j_password"])
        )
        driver.verify_connectivity()
        print("✅ Neo4j 连接成功")
    except Exception as e:
        print(f"❌ Neo4j 连接失败: {e}")
        return

    # 2. 从 ChromaDB 加载切片数据 (同时获取 collection 对象)
    # 注意：collection 名称必须和你之前构建时一致，可能是 'medqa_corpus' 或 'medqa_chunks'
    # 你可以去 Hybrid_RAG.py 里确认一下名字
    docs_text, collection = load_docs_from_chroma(CONFIG["chroma_path"], "medqa_corpus")

    if not docs_text or not collection:
        print("❌ 无法初始化系统，请检查向量库路径。")
        return

    # 3. 实例化核心系统
    try:
        print("⚙️ 正在组装系统组件...")
        app = MedicalQASystem(
            config=CONFIG,
            chroma_collection=collection,  # 传入 collection 对象
            neo4j_driver=driver,
            docs_text=docs_text,  # 传入从库里读出的切片列表
            table_path=CONFIG["table_path"]
        )
        print("\n🎉 系统初始化完成！可以开始提问了。")
    except Exception as e:
        print(f"❌ 系统初始化崩溃: {e}")
        return

    # 4. 交互循环
    print("-" * 50)
    print("输入 'exit' 退出")

    while True:
        try:
            query = input("\n👨‍⚕️ 问题: ").strip()
            if not query: continue
            if query.lower() in ["exit", "quit"]:
                break

            # 执行问答
            answer = app.run(query)

            print("\n🤖 回答:")
            print(answer)
            print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 出错: {e}")

    driver.close()


if __name__ == "__main__":
    main()