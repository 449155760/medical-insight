import pandas as pd
import json
import random
import os

# 建议用绝对路径或相对于脚本的路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(BASE_DIR, "../data/medical_data.xlsx")


def gen_data():
    if not os.path.exists(TABLE_PATH):
        print(f"❌ 错误：找不到文件 {TABLE_PATH}")
        return

    df = pd.read_excel(TABLE_PATH)
    questions = []

    # 1. 精确查询 (Retrieval) - RAG 和 Agent 都能做
    for _ in range(20):
        row = df.sample(1).iloc[0]
        questions.append({
            "type": "retrieval",
            "query": f"{row['药品名称']}的价格是多少？",
            "answer": str(row['价格(元)'])
        })

    # 2. 比较 (Comparison) - Agent 优势
    for _ in range(20):
        rows = df.sample(2)
        r1, r2 = rows.iloc[0], rows.iloc[1]
        truth = r1['药品名称'] if r1['价格(元)'] > r2['价格(元)'] else r2['药品名称']
        # 处理价格相等的情况
        if r1['价格(元)'] == r2['价格(元)']: truth = "一样贵"

        questions.append({
            "type": "comparison",
            "query": f"{r1['药品名称']}和{r2['药品名称']}哪个更贵？",
            "answer": truth
        })

    # 3. 聚合/计算 (Aggregation) - Agent 碾压 RAG 的关键
    # RAG 根本无法做这种题，Agent 写一行代码就搞定
    if '价格(元)' in df.columns:
        # 平均值
        avg_price = round(df['价格(元)'].mean(), 2)
        questions.append({
            "type": "aggregation",
            "query": "表中所有药品的平均价格是多少？",
            "answer": str(avg_price)
        })
        # 计数
        count = len(df[df['价格(元)'] > 50])
        questions.append({
            "type": "aggregation",
            "query": "价格超过50元的药品有多少种？",
            "answer": str(count)
        })

    output_path = os.path.join(BASE_DIR, "tabular_test_set.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"✅ 表格测试题生成完毕: {len(questions)} 条 -> {output_path}")


if __name__ == "__main__":
    gen_data()