import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 优先使用黑体显示中文，若无则回退到Arial
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def main():
    # 1. 文件路径
    file_path = '/home/lab207/data/ljy/Experiment/Tabular_Results_Fixed.xlsx'

    print(f"正在读取文件: {file_path}")
    try:
        # 读取 Excel 文件
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 数据处理
    # 统一 Type 列为小写并去除空格
    df['Type'] = df['Type'].str.lower().str.strip()

    # 按任务类型分组计算准确率均值
    # 确保列名正确，根据您之前提供的信息，列名应为 'RAG_Acc' 和 'Agent_Acc'
    if 'RAG_Acc' not in df.columns or 'Agent_Acc' not in df.columns:
        print("错误: 数据中缺少 'RAG_Acc' 或 'Agent_Acc' 列")
        return

    summary = df.groupby('Type')[['RAG_Acc', 'Agent_Acc']].mean()

    # 调整顺序：先 Retrieval 后 Comparison (符合一般逻辑：先简单后复杂)
    # 或者是按照 LaTeX 中的顺序
    desired_order = ['retrieval', 'comparison']
    summary = summary.reindex(desired_order)

    print("--- 准确率统计 ---")
    print(summary)

    # 3. 准备绘图数据
    labels = ['Retrieval (直接查值)', 'Comparison (数值对比)']  # 对应 LaTeX 中的 X 轴标签
    rag_scores = summary['RAG_Acc'].values
    agent_scores = summary['Agent_Acc'].values

    x = np.arange(len(labels))  # 标签位置
    width = 0.35  # 柱状图宽度

    # 4. 绘制柱状图
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 绘制两组柱子
    # 颜色参考 LaTeX: blue!30 (浅蓝) 和 red!30 (浅红)
    rects1 = ax.bar(x - width / 2, rag_scores, width, label='Markdown RAG', color='#a6cee3', edgecolor='blue',
                    alpha=0.8)
    rects2 = ax.bar(x + width / 2, agent_scores, width, label='Ours (Agent)', color='#fb9a99', edgecolor='red',
                    alpha=0.8)

    # 5. 添加文本标签、标题和图例
    ax.set_ylabel('准确率 (Accuracy)', fontsize=12)
    ax.set_title('表格问答任务准确率对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)  # 设置 Y 轴上限，留出空间显示数字
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=11)  # 图例放在底部

    # 添加网格线 (仅 Y 轴)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # 让网格线在柱子后面

    # 6. 在柱状图上方添加数值标签
    def autolabel(rects):
        """在每个柱子上方附加文本标签，显示其高度。"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移 3 点
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # 7. 保存和布局调整
    plt.tight_layout()
    output_file = 'tabular_qa_accuracy_bar.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"图表已保存至: {output_file}")

    # plt.show() # 如果在支持显示的终端运行，可取消注释


if __name__ == "__main__":
    main()