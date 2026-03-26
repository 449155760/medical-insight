import json
import random
import time
from tqdm import tqdm
from openai import OpenAI

# 配置你的 API
CONFIG = {
    "api_key": "sk-18cdce104dc64150a6039c86cef0fd4f",
    "base_url": "https://api.deepseek.com"
}

client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])


def generate_diverse_questions(intent, seed_examples, n=30):
    """利用 LLM 生成语义多样化的问题"""
    prompt = f"""你是一个医疗数据生成专家。请基于给定的意图类型和种子问题，生成 {n} 个语义多样化、口语化、长短不一的中文医疗问题。

    【意图类型】: {intent}
    【种子示例】: {seed_examples}

    【要求】:
    1. 模拟真实患者，包含焦虑语气、错别字或模糊描述。
    2. 不要只替换实体，要改变句式结构。
    3. 输出格式：纯 JSON 数组，例如 ["问题1", "问题2", ...]，不要 Markdown。
    """

    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8  # 高温度增加多样性
        )
        content = res.choices[0].message.content.strip()
        # 清洗 Markdown 标记
        content = content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        print(f"生成失败: {e}")
        return []


def main():
    print("🚀 正在生成高拟真测试数据...")

    dataset = []

    # 1. Comparison (对比类) - 这是你的创新点，多生成点
    seeds_comp = ["阿司匹林和布洛芬哪个副作用小？", "二甲双胍与胰岛素在治疗费用上的对比", "中药和西药哪个见效快"]
    print("生成 Comparison 数据...")
    questions = generate_diverse_questions("comparison (对比药物/疗法的优劣、价格、参数)", seeds_comp, n=40)
    for q in questions: dataset.append({"query": q, "label": "comparison"})

    # 2. Reasoning (推理类)
    seeds_reason = ["为什么二甲双胍会引起乳酸酸中毒？", "高热伴随颈强直可能是什么病？", "长期吃抗生素会有什么后果"]
    print("生成 Reasoning 数据...")
    questions = generate_diverse_questions("reasoning (需要复杂逻辑推理、病因分析、诊断)", seeds_reason, n=40)
    for q in questions: dataset.append({"query": q, "label": "reasoning"})

    # 3. Factual (事实类)
    seeds_fact = ["头痛吃什么药？", "糖尿病的定义是什么", "高血压诊断标准"]
    print("生成 Factual 数据...")
    questions = generate_diverse_questions("factual (单一事实查询、定义、标准)", seeds_fact, n=40)
    for q in questions: dataset.append({"query": q, "label": "factual"})

    # 打乱并保存
    random.shuffle(dataset)
    with open("router_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ 数据集生成完毕: router_benchmark.json (共 {len(dataset)} 条)")


if __name__ == "__main__":
    main()