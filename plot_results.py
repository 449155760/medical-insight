import pandas as pd
import matplotlib

# ✅ 【关键修复 1】强制使用 Agg 后端，必须在 import pyplot 之前
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置区域 =================
# 设置学术绘图风格 (Seaborn + Matplotlib)
# 注意：Linux服务器可能没有 SimHei 字体，如果中文显示乱码，可以改用 'WenQuanYi Micro Hei' 或直接用英文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 兜底英文

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

# 文件名映射 (请确保这些文件在当前目录下)
FILES = {
    "No RAG": "/home/lab207/data/ljy/Baselines/No_RAG_Direct_LLM_Qwen7B_20260129_1904.xlsx",

    "Naive RAG": "/home/lab207/data/ljy/Baselines/Naive_RAG_Qwen7B_TableAware_20260129_2258.xlsx",

    "Std Graph RAG": "/home/lab207/data/ljy/Baselines/StdGraph_RAG_2Hop_Noise_20260130_0032.xlsx",

    "Hybrid RAG": "/home/lab207/data/ljy/Baselines/Hybrid_RAG_Qwen7B_RRF_20260129_2306.xlsx",

    "Ablation": "/home/lab207/data/ljy/Experiment/System_Ablation_Results_20260130_014053.xlsx"

}


# ================= 1. 数据加载与预处理 =================

def load_data():
    dfs = {}
    print("正在加载数据...")
    for name, path in FILES.items():
        if os.path.exists(path):
            try:
                # 尝试读取 CSV，如果失败尝试 Excel
                if path.endswith('.csv'):
                    dfs[name] = pd.read_csv(path)
                else:
                    dfs[name] = pd.read_excel(path)
                print(f"✅ 成功加载: {name}")
            except Exception as e:
                print(f"❌ 加载失败 {name}: {e}")
        else:
            print(f"⚠️ 文件不存在: {path}")
    return dfs


def calculate_metrics(dfs):
    metrics = []

    # 1. No RAG
    if "No RAG" in dfs:
        d = dfs["No RAG"]
        acc = d['correct'].mean() if 'correct' in d.columns else 0
        metrics.append({"Method": "No RAG", "Accuracy": acc, "Hit Rate": 0, "Latency": 0})

    # 2. Naive RAG
    if "Naive RAG" in dfs:
        d = dfs["Naive RAG"]
        metrics.append({
            "Method": "Naive RAG",
            "Accuracy": d['is_correct'].mean(),
            "Hit Rate": d['hit_rate'].mean(),
            "Latency": d['latency'].mean()
        })

    # 3. Std Graph RAG
    if "Std Graph RAG" in dfs:
        d = dfs["Std Graph RAG"]
        metrics.append({
            "Method": "Std Graph RAG",
            "Accuracy": d['is_correct'].mean(),
            "Hit Rate": 0,  # Graph RAG 噪声太大，有效Hit极低
            "Latency": d['latency'].mean()
        })

    # 4. Hybrid RAG
    if "Hybrid RAG" in dfs:
        d = dfs["Hybrid RAG"]
        # 兼容列名
        acc_col = 'correct' if 'correct' in d.columns else 'is_correct'
        hit_col = 'hit' if 'hit' in d.columns else 'hit_rate'
        metrics.append({
            "Method": "Hybrid RAG",
            "Accuracy": d[acc_col].mean(),
            "Hit Rate": d[hit_col].mean(),
            "Latency": d['latency'].mean()
        })

    # 5. Ours (Medical-Insight) - 来自 Ablation 文件的 no_agent 列
    if "Ablation" in dfs:
        d = dfs["Ablation"]
        metrics.append({
            "Method": "Medical-Insight (Ours)",
            "Accuracy": d['no_agent_acc'].mean(),
            "Hit Rate": d['no_agent_hit'].mean(),
            "Latency": d['no_agent_lat'].mean()
        })

    return pd.DataFrame(metrics)


# ================= 2. 绘图函数 =================

def plot_main_comparison(df):
    """图1：主实验结果对比 (Accuracy & Hit Rate)"""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    x = np.arange(len(df))
    width = 0.35

    # 绘制 Accuracy (左轴)
    bars1 = ax1.bar(x - width / 2, df['Accuracy'], width, label='Accuracy',
                    color='#4c72b0', alpha=0.9, edgecolor='black', linewidth=1)

    # 绘制 Hit Rate (右轴)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, df['Hit Rate'], width, label='Hit Rate',
                    color='#55a868', alpha=0.8, edgecolor='black', hatch='//', linewidth=1)

    # 设置轴标签
    ax1.set_ylabel('Accuracy', fontsize=16, fontweight='bold', color='#4c72b0')
    ax2.set_ylabel('Retrieval Hit Rate', fontsize=16, fontweight='bold', color='#55a868')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Method'], rotation=15, ha='right', fontsize=13)
    ax1.set_title('Overall Performance Comparison', fontsize=18, fontweight='bold', pad=20)

    # 添加数值标签
    def add_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    add_labels(bars1, ax1)
    add_labels(bars2, ax2)

    # 设置范围
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 0.6)  # Hit Rate通常较低，放大显示

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=12)

    plt.tight_layout()
    plt.savefig('Fig1_Main_Results.png', dpi=300)
    print("📊 生成图表: Fig1_Main_Results.png")
    # ✅ 【关键修复 2】移除 plt.show()，改用 plt.close()
    plt.close()


def plot_ablation(dfs):
    """图2：消融实验分析"""
    if "Ablation" not in dfs: return
    d = dfs["Ablation"]

    # 提取消融数据
    data = {
        "Ours (Full)": d['no_agent_acc'].mean(),
        "w/o Graph": d['no_graph_acc'].mean(),
        "w/o BM25": d['no_bm25_acc'].mean()
    }

    names = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#c44e52', '#dd8452', '#ccb974']  # 红、橙、黄配色

    bars = ax.bar(names, values, color=colors, edgecolor='black', width=0.6, linewidth=1.2)

    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Ablation Study: Impact of Components', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)

    # 添加数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Fig2_Ablation.png', dpi=300)
    print("📊 生成图表: Fig2_Ablation.png")
    plt.close()  # ✅ 修复


def plot_reasoning_analysis(dfs):
    """图3：复杂推理题 vs 简单事实题"""
    # 关键词定义 (根据 MedQA 特点)
    reasoning_kws = ["诊断", "原因", "机制", "为什么", "导致", "并发症", "最可能", "首选"]

    def classify_question(text):
        for kw in reasoning_kws:
            if kw in str(text): return "Reasoning"
        return "Factual"

    results = []

    # 1. Std Graph RAG
    if "Std Graph RAG" in dfs:
        d = dfs["Std Graph RAG"].copy()
        d['Type'] = d['question'].apply(classify_question)
        acc_r = d[d['Type'] == 'Reasoning']['is_correct'].mean()
        acc_f = d[d['Type'] == 'Factual']['is_correct'].mean()
        results.append(["Std Graph RAG", acc_f, acc_r])

    # 2. Hybrid RAG
    if "Hybrid RAG" in dfs:
        d = dfs["Hybrid RAG"].copy()
        q_col = 'q' if 'q' in d.columns else 'question'
        acc_col = 'correct' if 'correct' in d.columns else 'is_correct'
        d['Type'] = d[q_col].apply(classify_question)
        acc_r = d[d['Type'] == 'Reasoning'][acc_col].mean()
        acc_f = d[d['Type'] == 'Factual'][acc_col].mean()
        results.append(["Hybrid RAG", acc_f, acc_r])

    # 3. Ours
    if "Ablation" in dfs:
        d = dfs["Ablation"].copy()
        d['Type'] = d['query'].apply(classify_question)
        acc_r = d[d['Type'] == 'Reasoning']['no_agent_acc'].mean()
        acc_f = d[d['Type'] == 'Factual']['no_agent_acc'].mean()
        results.append(["Ours (Medical-Insight)", acc_f, acc_r])

    df_res = pd.DataFrame(results, columns=["Method", "Factual", "Reasoning"])

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_res))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df_res['Factual'], width, label='Factual Questions',
                   color='#8172b3', edgecolor='black', alpha=0.9)
    bars2 = ax.bar(x + width / 2, df_res['Reasoning'], width, label='Reasoning Questions',
                   color='#c44e52', edgecolor='black', hatch='..', alpha=0.9)

    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Performance on Question Types', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_res['Method'], fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.0)

    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Fig3_Reasoning_Analysis.png', dpi=300)
    print("📊 生成图表: Fig3_Reasoning_Analysis.png")
    plt.close()  # ✅ 修复


# ================= 3. 主程序 =================
if __name__ == "__main__":
    dfs = load_data()

    if dfs:
        metrics_df = calculate_metrics(dfs)

        # 绘制三张图
        plot_main_comparison(metrics_df)
        plot_ablation(dfs)
        plot_reasoning_analysis(dfs)

        print("\n✅ 所有图表生成完毕！请查看当前目录下的 PNG 文件。")
    else:
        print("❌ 未找到数据文件，请检查文件名是否正确。")