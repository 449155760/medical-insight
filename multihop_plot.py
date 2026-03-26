import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格 (可选)
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 用来正常显示中文标签，如果环境不支持可注释掉
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def main():
    # 1. 文件路径配置
    file_path = '/home/lab207/data/ljy/Experiment/Multihop_Hybrid_Results_20260109_1715.xlsx'

    # 2. 读取数据
    print(f"正在读取文件: {file_path}")
    try:
        # 读取 Excel 文件，默认读取第一个 Sheet
        df = pd.read_excel(file_path, engine='openpyxl')
    except FileNotFoundError:
        print("错误: 未找到文件，请检查路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 3. 数据处理：按 Hops 分组计算准确率均值
    # 确保列名存在，如果实际列名有出入，请在此处修改
    required_columns = ['Hops', 'Vector_Acc', 'KGQA_Acc', 'Ours_Acc']

    # 检查列名是否匹配
    if not all(col in df.columns for col in required_columns):
        print(f"错误: 数据表中缺少必要的列。需要包含: {required_columns}")
        print(f"实际列名: {df.columns.tolist()}")
        return

    # 计算均值
    df_grouped = df.groupby('Hops')[['Vector_Acc', 'KGQA_Acc', 'Ours_Acc']].mean()
    print("--- 计算结果 (Accuracy Mean) ---")
    print(df_grouped)

    # 4. 绘图 (折线图)
    plt.figure(figsize=(10, 6), dpi=300)

    # 定义线条样式
    # Vector RAG: 蓝色，三角形，虚线
    plt.plot(df_grouped.index, df_grouped['Vector_Acc'],
             color='#1f77b4', marker='^', linestyle='--', linewidth=2, markersize=8, label='Vector RAG')

    # Standard KGQA: 红色，正方形，点划线
    plt.plot(df_grouped.index, df_grouped['KGQA_Acc'],
             color='#d62728', marker='s', linestyle='-.', linewidth=2, markersize=8, label='Standard KGQA')

    # Ours (Vector-ToG): 绿色，圆形，实线
    plt.plot(df_grouped.index, df_grouped['Ours_Acc'],
             color='#2ca02c', marker='o', linestyle='-', linewidth=2.5, markersize=9, label='Ours (Vector-ToG)')

    # 5. 图表装饰
    plt.title('Performance Comparison across Reasoning Hops', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Reasoning Depth (Hops)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    # 设置 X 轴刻度为整数 (1, 2, 3)
    plt.xticks([1, 2, 3], ['1-Hop', '2-Hop', '3-Hop'], fontsize=12)
    plt.yticks(fontsize=12)

    # 设置 Y 轴范围 (0 到 1.05)
    plt.ylim(-0.05, 1.05)

    # 添加图例
    plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 6. 保存图片
    output_file = 'multihop_accuracy_trend.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"图表已保存至: {output_file}")

    # 显示图表 (如果在支持图形界面的环境中运行)
    # plt.show()


if __name__ == "__main__":
    main()