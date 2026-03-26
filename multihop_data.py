import json
import random
from neo4j import GraphDatabase
from tqdm import tqdm
from openai import OpenAI

# ================= 配置 =================
CONFIG = {
    "api_key": "sk-18cdce104dc64150a6039c86cef0fd4f",
    "base_url": "https://api.deepseek.com",
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "Ljy449155760",
    "output_file": "Multihop_data.json"
}

client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])


def paraphrase_question(template_q, answer, hops):
    """LLM 润色问题"""
    difficulty = "中等" if hops < 3 else "困难"
    # 将列表答案转为字符串用于提示，防止LLM困惑
    ans_str = str(answer) if isinstance(answer, list) else answer

    prompt = f"""
    你是一个医学考试出题专家。请将以下逻辑简单的“模板问题”改写为一个“{difficulty}”的临床情境题。

    模板原题: "{template_q}"
    标准答案范围: "{ans_str}"

    要求:
    1. 重新组织语言，模拟真实患者描述或医生查房。
    2. 不要改变核心医学事实。
    3. 确保问题指向标准答案（或其所属类别）。
    4. 直接输出问题，不要加引号。
    """
    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content.strip()
    except:
        return template_q


def generate_hop_questions(driver, hops=1, count=30):
    questions = []
    print(f"🔄 正在挖掘 {hops}-Hop 路径...")

    with driver.session() as session:
        if hops == 1:
            # 1-Hop: 聚合所有症状，防止“答案不唯一”导致的误判
            query = """
            MATCH (a:Disease)-[:has_symptom]->(b:Symptom) 
            WITH a, collect(b.name) as symptoms
            WHERE size(symptoms) >= 1
            RETURN a.name, symptoms LIMIT 200
            """
            res = session.run(query).data()
            if res:
                samples = random.sample(res, min(len(res), count))
                for s in tqdm(samples, desc="1-Hop Generating"):
                    # 随机选一个症状放入问题模板，但Truth保留所有症状
                    target_symptom = s['symptoms'][0]
                    raw_q = f"{s['a.name']}通常会出现什么症状？"

                    questions.append({
                        "query": raw_q,  # 1-Hop 简单改写即可，或者直接用
                        "answer": ", ".join(s['symptoms']),  # 关键：Ground Truth 包含所有正确答案
                        "hops": 1,
                        "type": "Direct"
                    })

        elif hops == 2:
            # 2-Hop: A并发B -> C症状
            query = """
            MATCH (a:Disease)-[:acompany_with]->(b:Disease)-[:has_symptom]->(c:Symptom)
            WITH a, b, collect(c.name) as symptoms
            WHERE size(symptoms) >= 1
            RETURN a.name, b.name, symptoms LIMIT 300
            """
            res = session.run(query).data()
            if res:
                samples = random.sample(res, min(len(res), count))
                for s in tqdm(samples, desc="2-Hop Generating"):
                    # 取其中一个症状作为改写目标，但答案包含所有
                    main_symptom = s['symptoms'][0]
                    raw_q = f"患有{s['a.name']}的病人如果并发了{s['b.name']}，通常会出现什么新症状（例如{main_symptom}）？"
                    polished_q = paraphrase_question(f"患有{s['a.name']}的病人如果并发了{s['b.name']}，会有什么表现？",
                                                     main_symptom, 2)

                    questions.append({
                        "query": polished_q,
                        "answer": ", ".join(s['symptoms']),  # Ground Truth 包含所有
                        "hops": 2,
                        "type": "Complication"
                    })

        elif hops == 3:
            # 3-Hop: 药物反推疾病
            # 这里答案通常是唯一的（特定的疾病C），所以不需要collect
            query = """
            MATCH (a:Disease)-[:recommand_drug]->(b:Drug)<-[:recommand_drug]-(c:Disease)-[:has_symptom]->(d:Symptom)
            WHERE a.name <> c.name
            RETURN a.name, b.name, c.name, d.name LIMIT 300
            """
            res = session.run(query).data()
            if len(res) > 0:
                samples = random.sample(res, min(len(res), count))
                for s in tqdm(samples, desc="3-Hop Generating"):
                    raw_q = f"用于治疗{s['a.name']}的{s['b.name']}，也是另一种会出现{s['d.name']}症状的疾病的常用药，那种病是什么？"
                    polished_q = paraphrase_question(raw_q, s['c.name'], 3)

                    questions.append({
                        "query": polished_q,
                        "answer": s['c.name'],  # 3-Hop 答案通常是单实体
                        "hops": 3,
                        "type": "Drug_Interaction"
                    })

    return questions


def main():
    try:
        driver = GraphDatabase.driver(CONFIG["neo4j_uri"], auth=(CONFIG["neo4j_user"], CONFIG["neo4j_password"]))
        dataset = []

        # 生成数据
        dataset.extend(generate_hop_questions(driver, 1, 20))  # 每类生成20条，总共60条足够评测
        dataset.extend(generate_hop_questions(driver, 2, 20))
        dataset.extend(generate_hop_questions(driver, 3, 20))

        # 保存
        with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 数据生成完毕: {CONFIG['output_file']}，共 {len(dataset)} 条。")
        driver.close()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()