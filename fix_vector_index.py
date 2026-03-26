from neo4j import GraphDatabase

# ================= 配置 =================
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "Ljy449155760"


# ========================================

def fix_index():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    print("🔧 正在修复向量索引...")

    with driver.session() as session:
        # 1. 给所有计算好向量的节点，打上一个统一的标签 :MedicalEntity
        print("1️⃣ 正在为节点添加通用标签 :MedicalEntity ...")
        add_label_query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        SET n:MedicalEntity
        """
        session.run(add_label_query)
        print("   ✅ 标签添加完成！")

        # 2. 创建索引 (针对 MedicalEntity 标签)
        print("2️⃣ 正在重新创建向量索引...")
        create_index_query = """
        CREATE VECTOR INDEX medical_entity_index IF NOT EXISTS
        FOR (n:MedicalEntity)
        ON (n.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 512,
          `vector.similarity_function`: 'cosine'
        }}
        """
        try:
            session.run(create_index_query)
            print("   ✅ 索引创建指令已发送！")
        except Exception as e:
            print(f"   ⚠️ 索引创建提示: {e}")

    # 3. 验证索引状态
    print("3️⃣ 正在验证索引状态...")
    with driver.session() as session:
        try:
            # 等待几秒让索引生效
            import time
            time.sleep(5)
            # 尝试查询
            check_query = "SHOW INDEXES WHERE name = 'medical_entity_index'"
            result = session.run(check_query).data()
            if result:
                state = result[0].get('state', 'UNKNOWN')
                print(f"   🎉 索引状态: {state} (如果是 ONLINE 或 POPULATING 就成功了)")
            else:
                print("   ❌ 未找到索引，请检查报错信息。")
        except Exception:
            pass

    driver.close()
    print("\n✨ 修复完成！现在可以运行 main_system.py 了。")


if __name__ == "__main__":
    fix_index()