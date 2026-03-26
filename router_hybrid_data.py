import json
import random
import os
import pandas as pd

# ================= 配置区域 =================
CONFIG = {
    # 你的 CMID 绝对路径
    "cmid_path": r"/home/lab207/data/ljy/data/CMID.json",
    # 之前生成的合成数据路径
    "synthetic_path": "router_benchmark.json",
    # 输出文件
    "output_path": "router_hybrid_benchmark.json",

    # 【科学性关键】采样数量设置 (参考你的论文大纲)
    "sample_size": {
        "factual": 60,  # 从 CMID 抽取
        "reasoning": 60,  # 从 CMID 抽取
        "comparison": 60  # 从 Synthetic 抽取 (因为 CMID 几乎没有对比类)
    }
}

# ================= 1. 定义映射规则 (Taxonomy Mapping) =================
# 将 CMID 的 36 类标签映射到你的 3 大类
# 科学性依据：依据语义意图的复杂度和所需推理深度划分
LABEL_MAP = {
    # --- Factual (事实类：查定义、标准、属性) ---
    "定义": "factual",
    "病症": "factual",
    "临床表现": "factual",
    "治疗方法": "factual",
    "所属科室": "factual",
    "传染性": "factual",
    "治愈率": "factual",
    "禁忌": "factual",
    "上市时间": "factual",
    "用法用量": "factual",

    # --- Reasoning (推理类：查原因、机制、诊断) ---
    "病因分析": "reasoning",
    "严重性": "reasoning",  # 通常需要综合判断
    "预防": "reasoning",  # 涉及机制推导
    "作用机制": "reasoning",  # 强推理

    # --- Comparison (CMID 中几乎没有，这里作为占位) ---
    "多问": "comparison",  # 勉强算，但通常质量不高
    "对比": "comparison"
}


def load_cmid(filepath):
    """读取 CMID 数据并进行映射"""
    data_pool = {"factual": [], "reasoning": [], "comparison": []}

    if not os.path.exists(filepath):
        print(f"❌ 错误：找不到 CMID 文件: {filepath}")
        return data_pool

    try:
        # CMID 通常是一行一个 JSON 或者 整个是一个 JSON 列表
        # 这里假设是标准 JSON 列表格式
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        print(f"📂 正在处理 CMID 原数据 ({len(raw_data)} 条)...")

        for item in raw_data:
            # 获取原始 query
            query = item.get("originalText") or item.get("text")
            # 获取原始标签 (CMID 可能是列表)
            raw_label = item.get("label_36class")
            if isinstance(raw_label, list): raw_label = raw_label[0]

            # 进行映射
            target_label = LABEL_MAP.get(raw_label)

            if target_label and query:
                data_pool[target_label].append({
                    "query": query,
                    "label": target_label,
                    "source": "Real (CMID)"  # 标记来源，方便后续分析
                })

        return data_pool
    except Exception as e:
        print(f"⚠️ 读取 CMID 失败: {e}")
        return data_pool


def load_synthetic(filepath):
    """读取合成数据"""
    data_pool = {"factual": [], "reasoning": [], "comparison": []}
    if not os.path.exists(filepath): return data_pool

    with open(filepath, "r", encoding="utf-8") as f:
        items = json.load(f)
        for item in items:
            lbl = item["label"]
            item["source"] = "Synthetic (LLM)"
            if lbl in data_pool:
                data_pool[lbl].append(item)
    return data_pool


def main():
    print("🚀 开始构建混合基准数据集 (Hybrid Benchmark)...")

    # 1. 加载数据
    cmid_pool = load_cmid(CONFIG["cmid_path"])
    syn_pool = load_synthetic(CONFIG["synthetic_path"])

    final_dataset = []
    stats = {"factual": 0, "reasoning": 0, "comparison": 0}

    # 2. 分层采样 (Stratified Sampling)
    # 策略：Factual/Reasoning 优先用 CMID (真实)，Comparison 用 Synthetic (合成)

    # --- 构建 Factual ---
    # 优先从 CMID 随机抽
    needed = CONFIG["sample_size"]["factual"]
    if len(cmid_pool["factual"]) >= needed:
        samples = random.sample(cmid_pool["factual"], needed)
    else:
        # 如果 CMID 不够，用合成数据补（通常 CMID 足够）
        print("⚠️ CMID Factual 数据不足，使用合成数据补全")
        samples = cmid_pool["factual"] + syn_pool["factual"][:needed - len(cmid_pool["factual"])]
    final_dataset.extend(samples)
    stats["factual"] = len(samples)

    # --- 构建 Reasoning ---
    needed = CONFIG["sample_size"]["reasoning"]
    if len(cmid_pool["reasoning"]) >= needed:
        samples = random.sample(cmid_pool["reasoning"], needed)
    else:
        samples = cmid_pool["reasoning"] + syn_pool["reasoning"][:needed - len(cmid_pool["reasoning"])]
    final_dataset.extend(samples)
    stats["reasoning"] = len(samples)

    # --- 构建 Comparison ---
    # CMID 极度缺乏此类，直接使用合成数据
    needed = CONFIG["sample_size"]["comparison"]
    samples = random.sample(syn_pool["comparison"], min(len(syn_pool["comparison"]), needed))
    final_dataset.extend(samples)
    stats["comparison"] = len(samples)

    # 3. 打乱并保存
    random.shuffle(final_dataset)

    with open(CONFIG["output_path"], "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 混合数据集构建完成！保存至: {CONFIG['output_path']}")
    print(f"📊 数据分布统计: {stats}")
    print(f"   总计: {len(final_dataset)} 条")

    # 预览
    print("\n🔍 数据预览 (前3条):")
    print(json.dumps(final_dataset[:3], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()