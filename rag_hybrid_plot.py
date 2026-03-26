import pandas as pd

# 1. 读取数据
df = pd.read_json("/home/lab207/data/ljy/Baselines/Hybrid_RAG_Enhanced_20260114_1900.jsonl", lines=True)

# 2. 计算准确率 (全量)
accuracy = df['correct'].mean()
print(f"Accuracy: {accuracy:.2%}")

# 3. 计算检索命中率 (全量)
hit_rate = df['hit'].mean()
print(f"Hit Rate: {hit_rate:.2%}")

# 4. 计算忠实度/相关性 (⚠️ 关键：必须过滤掉 0 值)
# 因为代码里只采样了 10%，剩下的 0 是没测的，不能算进平均分
sampled_df = df[df['faithfulness'] > 0]
avg_faith = sampled_df['faithfulness'].mean()
avg_rel = sampled_df['relevance'].mean()

print(f"Avg Faithfulness: {avg_faith:.2f} (Sample Size: {len(sampled_df)})")
print(f"Avg Relevance: {avg_rel:.2f}")