import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置区域 =================
FILE_PATH = "/home/lab207/data/ljy/Experiment/Router_Experiment_Results_20260108_212127.xlsx"
SAVE_DIR = os.path.dirname(FILE_PATH) if os.path.exists(os.path.dirname(FILE_PATH)) else "."

# 统一色卡 (学术风格)
PALETTE = {
    'Regex': '#7f8c8d',  # 灰色 (Asbestos)
    'LLM': '#c0392b',  # 红色 (Pomegranate)
    'Embedding': '#2980b9',  # 蓝色 (Belize Hole)
    'Ours': '#27ae60'  # 绿色 (Nephritis)
}

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False


def load_data(path):
    if path.endswith('.csv'): return pd.read_csv(path)
    return pd.read_excel(path)


def plot_tradeoff(df):
    """绘制准确率 vs 延迟的权衡散点图"""
    print("📊 正在绘制 Trade-off 散点图...")

    summary = df.groupby('Strategy').agg({'Correct': 'mean', 'Latency': 'mean'}).reset_index()

    name_map = {
        'A: Regex Only': 'Strategy A\n(Regex)',
        'B: Full LLM': 'Strategy B\n(Full LLM)',
        'C: Embedding Proto': 'Strategy C\n(Embedding)',
        'Ours: Cascade': 'Ours\n(Cascade Router)'
    }
    summary['Strategy_Label'] = summary['Strategy'].map(name_map)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 绘制点
    for i, row in summary.iterrows():
        lbl = row['Strategy_Label']
        # 动态分配样式
        if "Regex" in lbl:
            c, m, s = PALETTE['Regex'], 's', 150
        elif "Full LLM" in lbl:
            c, m, s = PALETTE['LLM'], '^', 150
        elif "Embedding" in lbl:
            c, m, s = PALETTE['Embedding'], 'D', 150
        elif "Ours" in lbl:
            c, m, s = PALETTE['Ours'], '*', 400  # Ours 最大

        ax.scatter(row['Latency'], row['Correct'], c=c, marker=m, s=s,
                   label=row['Strategy'].split(':')[0], edgecolors='white', linewidth=1.5, zorder=10)

        # 标签微调
        offset_y = 0.008
        if "Regex" in lbl: offset_y = -0.015
        ax.text(row['Latency'], row['Correct'] + offset_y,
                f"{lbl}\n({row['Correct']:.3f})",
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=c)

    # 设置坐标轴
    ax.set_xscale('log')
    ax.set_xlabel('Average Latency (ms) [Log Scale]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency vs. Accuracy Trade-off', fontsize=14, pad=20)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # 【改进】聚焦 Y 轴
    ax.set_ylim(0.50, 0.70)

    # 【改进】辅助线与注释
    ours = summary[summary['Strategy'].str.contains("Ours")].iloc[0]
    llm = summary[summary['Strategy'].str.contains("Full LLM")].iloc[0]

    # 虚线连接
    ax.plot([ours['Latency'], llm['Latency']], [ours['Correct'], llm['Correct']],
            color='gray', linestyle='--', alpha=0.6, zorder=1)
    ax.text((ours['Latency'] * llm['Latency']) ** 0.5, ours['Correct'] + 0.002,
            "Lossless Accuracy", ha='center', va='bottom', fontsize=9, style='italic', color='gray')

    # 最优区箭头
    ax.annotate('Ideal Zone\n(Fast & Accurate)', xy=(0.1, 0.68), xycoords='data',
                xytext=(0.5, 0.68), textcoords='data',
                arrowprops=dict(facecolor='black', arrowstyle="->", alpha=0.5),
                fontsize=9, ha='right', alpha=0.7)

    save_path = os.path.join(SAVE_DIR, 'router_tradeoff_v2.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Trade-off 图已保存: {save_path}")


def plot_traffic_distribution(df):
    """绘制流量分流环形图"""
    print("🥧 正在绘制流量分流图...")

    ours_df = df[df['Strategy'] == 'Ours: Cascade'].copy()

    def infer_level(cost):
        if cost < 0.01:
            return "L1: Regex Match"  # 对应 Strategy A
        elif cost < 0.8:
            return "L2: Embedding Filter"  # 对应 Strategy C
        else:
            return "L3: LLM Reasoning"  # 对应 Strategy B

    ours_df['Route_Level'] = ours_df['Cost'].apply(infer_level)
    level_counts = ours_df['Route_Level'].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # 【改进】颜色映射与 Scatter 图一致
    color_map = {
        "L1: Regex Match": PALETTE['Regex'],
        "L2: Embedding Filter": PALETTE['Embedding'],
        "L3: LLM Reasoning": PALETTE['LLM']
    }
    plot_colors = [color_map.get(idx, 'gray') for idx in level_counts.index]

    wedges, texts, autotexts = ax.pie(
        level_counts, labels=level_counts.index, autopct='%1.1f%%',
        startangle=140, colors=plot_colors, pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='white'),
        textprops=dict(color="black")
    )

    plt.setp(texts, size=11, fontweight='bold')
    plt.setp(autotexts, size=10, weight="bold", color="white")

    # 中心文字
    offloaded_pct = 100 - (level_counts.get("L3: LLM Reasoning", 0) / len(ours_df) * 100)
    ax.text(0, 0, f"Total\nOffloaded\n{offloaded_pct:.1f}%",
            ha='center', va='center', fontsize=12, fontweight='bold', color=PALETTE['Ours'])

    ax.set_title("Traffic Offloading Distribution (Ours)", fontsize=14, pad=20)

    save_path = os.path.join(SAVE_DIR, 'router_traffic_pie_v2.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 饼图已保存: {save_path}")


if __name__ == "__main__":
    df = load_data(FILE_PATH)
    plot_tradeoff(df)
    plot_traffic_distribution(df)