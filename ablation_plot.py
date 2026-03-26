import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 设置学术风格的绘图参数
plt.style.use('seaborn-v0_8-paper')  # 使用论文风格
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体 (Times New Roman风格)
plt.rcParams['axes.linewidth'] = 1.5  # 边框加粗


def plot_ablation_study(file_path):
    # 2. 读取数据
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return

    # 3. 数据处理：计算准确率和平均延迟
    # 映射模式名称到论文中使用的标准名称
    name_mapping = {
        'full': 'Full System',
        'no_routing': 'w/o Routing',
        'no_graph': 'w/o Graph',
        'no_agent': 'w/o Agent'
    }

    # 确保只处理存在的模式
    df = df[df['mode'].isin(name_mapping.keys())].copy()

    summary = df.groupby('mode').agg({
        'correct': 'mean',  # 准确率
        'latency': 'mean'  # 平均延迟
    }).reset_index()

    # 应用映射并排序 (Full System 排在第一位)
    summary['mode_display'] = summary['mode'].map(name_mapping)

    # 自定义排序逻辑：Full System -> w/o Routing -> w/o Graph -> w/o Agent
    sort_order = ['Full System', 'w/o Routing', 'w/o Graph', 'w/o Agent']
    summary['mode_display'] = pd.Categorical(summary['mode_display'], categories=sort_order, ordered=True)
    summary = summary.sort_values('mode_display')

    # 4. 开始绘图
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # X轴位置
    x = np.arange(len(summary))
    width = 0.4

    # --- 左轴：准确率 (柱状图) ---
    color_acc = '#4C72B0'  # 经典的学术蓝
    bars = ax1.bar(x, summary['correct'], width, color=color_acc, label='Accuracy', alpha=0.9, edgecolor='black')

    # 设置左轴标签和范围
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color=color_acc)
    ax1.set_ylim(0.88, 0.94)  # 缩放Y轴以突显差异 (根据你的数据范围0.89-0.92调整)
    ax1.tick_params(axis='y', labelcolor=color_acc)

    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- 右轴：延迟 (折线图) ---
    ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴
    color_lat = '#C44E52'  # 经典的学术红

    # 绘制折线 + 标记点
    line = ax2.plot(x, summary['latency'], color=color_lat, marker='o',
                    linewidth=2.5, markersize=8, label='Avg Latency (ms)', linestyle='--')

    # 设置右轴标签和范围
    ax2.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold', color=color_lat)
    ax2.set_ylim(8, 16)  # 根据数据范围调整 (9ms - 15ms)
    ax2.tick_params(axis='y', labelcolor=color_lat)

    # 在折线点上添加数值标签
    for i, txt in enumerate(summary['latency']):
        # 特殊处理：把 w/o Routing 的标签放高一点，避免挡住线
        offset = 0.5 if txt > 14 else -0.8
        ax2.text(x[i], txt + offset, f'{txt:.1f} ms',
                 ha='center', va='bottom', color=color_lat, fontsize=10, fontweight='bold')

    # 5. 格式化图表
    ax1.set_xlabel('Module Ablation Settings', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary['mode_display'], fontsize=11)

    plt.title('Ablation Study: Performance vs. Efficiency Trade-off', fontsize=14, pad=20)

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    plt.tight_layout()

    # 6. 保存图片
    output_filename = 'ablation_study_dual_axis.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"✅ 图表已生成并保存为: {output_filename}")
    plt.show()


# 运行函数
if __name__ == "__main__":
    file_path = '/home/lab207/data/ljy/Experiment/ablation_results_20251228_023036.csv'
    plot_ablation_study(file_path)