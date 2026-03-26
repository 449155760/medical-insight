import subprocess
import time
import os
import sys

# 设置 HuggingFace 国内镜像 (全局生效)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 定义要运行的脚本列表 (按顺序)
SCRIPTS = [

    # 3. Naive RAG 基线 - 约 1小时
    "/home/lab207/data/ljy/Baselines/Naive_RAG.py",

    # 4. Hybrid RAG 基线 - 约 1.5小时 x
    "/home/lab207/data/ljy/Baselines/Hybrid_RAG.py",

    # 5. Standard Graph RAG 基线 - 约 2小时
    "/home/lab207/data/ljy/Baselines/StdGraph_RAG.py"
    
# 1. 主系统全量评测 (含 LLM 打分，用于和基线对比) - 约 2小时
    "/home/lab207/data/ljy/Experiment/Ours_evaluate.py",

]

def run_script(script_name):
    """运行单个脚本，并等待其结束"""
    print(f"\n{'=' * 60}")
    print(f"🚀 正在启动: {script_name}")
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # 获取当前 Python解释器路径
    python_exe = sys.executable

    try:
        # 使用 subprocess 串行调用
        result = subprocess.run([python_exe, script_name], check=True)
        print(f"\n✅ {script_name} 运行完成!")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} 运行出错! (Exit Code: {e.returncode})")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")

    # 冷却时间，让显卡显存完全释放
    print("❄️ 冷却 30秒，释放 GPU 资源...")
    time.sleep(30)


if __name__ == "__main__":
    total_start = time.time()
    print("🏁 开始全量基线评测任务 (Auto-Runner)")
    print(f"📂 工作目录: {os.getcwd()}")

    for script in SCRIPTS:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"⚠️ 警告: 找不到文件 {script}，跳过。")

    total_duration = time.time() - total_start
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)

    print(f"\n{'=' * 60}")
    print(f"🎉 所有任务执行完毕！")
    print(f"⏱️ 总耗时: {hours}小时 {minutes}分钟")
    print(f"💾 请检查 runs/ 和 logs/ 目录下的结果文件")
    print(f"{'=' * 60}")