import pandas as pd
import os

# ================= 配置区域 =================
# 定义文件路径映射
# 请确保文件名与你实际硬盘上的文件名一致
FILES = {
    "Naive RAG": "D:/pycharm/pythonProject3/Baselines/Naive_RAG_Result_20251228_0847.xlsx",
    "Hybrid RAG": "D:/pycharm/pythonProject3/Baselines/Hybrid_RAG_Result_20260108_1645.xlsx",
    "Std GraphRAG": "D:/pycharm/pythonProject3/Baselines/StdGraph_RAG_Result_20251228_0900.xlsx",
    "Ours (Proposed)": "D:/pycharm/pythonProject3/Experiment/Ours_System_Evaluation_20260109_1859.xlsx"
}


def analyze_results():
    print(f"\n{'Method':<20} | {'Acc':<8} | {'Rel':<8} | {'Faith':<8} | {'Overall':<8}")
    print("-" * 68)

    results = {}

    for method, filepath in FILES.items():
        # 1. 检查文件是否存在
        if not os.path.exists(filepath):
            print(f"⚠️ 文件未找到: {filepath} (跳过)")
            continue

        try:
            # 读取 Excel
            df = pd.read_excel(filepath)

            # ================= 指标计算 =================

            # 1. 准确率 (Accuracy)
            # 只要有 is_correct 列就能算
            if 'is_correct' in df.columns:
                acc = df['is_correct'].mean()
            else:
                acc = 0.0
                print(f"   Warning: {method} 缺少 'is_correct' 列")

            # 2. 相关性 (Relevance)
            # 检查列是否存在，如果不存在则给 0，防止报错
            if 'relevance' in df.columns:
                raw_rel = df['relevance'].mean()
                # 如果是NaN（因为是抽样评测），mean()会自动忽略NaN，结果可能为nan
                if pd.isna(raw_rel): raw_rel = 0.0

                # 归一化处理 (假设原分是5分制)
                rel = raw_rel / 5.0 if raw_rel > 1 else raw_rel
            else:
                rel = 0.0  # 缺少数据时默认为 0

            # 3. 忠实度 (Faithfulness)
            if 'faithfulness' in df.columns:
                raw_faith = df['faithfulness'].mean()
                if pd.isna(raw_faith): raw_faith = 0.0
                faith = raw_faith / 5.0 if raw_faith > 1 else raw_faith
            else:
                faith = 0.0  # 缺少数据时默认为 0

            # 4. 综合得分
            # 如果 Rel 和 Faith 为 0，综合得分只看 Acc 吗？这里还是除以 3，保持公式一致
            overall = (acc + rel + faith) / 3.0

            # ================= 输出结果 =================
            results[method] = {
                "Acc": acc,
                "Rel": rel,
                "Faith": faith,
                "Overall": overall
            }

            print(f"{method:<20} | {acc:.4f}   | {rel:.4f}   | {faith:.4f}   | {overall:.4f}")

            # 如果缺失关键指标，打印提示
            missing_cols = []
            if 'relevance' not in df.columns: missing_cols.append('relevance')
            if 'faithfulness' not in df.columns: missing_cols.append('faithfulness')
            if missing_cols:
                print(f"   └── ⚠️ 警告: 该文件缺少列 {missing_cols}，相关指标显示为 0.0000")

        except Exception as e:
            print(f"❌ 处理 {method} 时发生未知错误: {e}")

    return results


if __name__ == "__main__":
    analyze_results()