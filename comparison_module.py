import pandas as pd
from openai import OpenAI
import re


class ComparisonModule:
    """
    表格分析模块：Text-to-Pandas Agent
    """

    def __init__(self, config, table_path):
        self.config = config
        self.client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        try:
            if table_path.endswith('.csv'):
                self.df = pd.read_csv(table_path)
            else:
                self.df = pd.read_excel(table_path)
        except:
            self.df = pd.DataFrame()  # 空表处理

    def query_table(self, query):
        if self.df.empty: return "数据表未加载或为空。"

        # 提示词：让 Qwen 生成 Python 代码
        prompt = f"""你是一个Python数据分析助手。DataFrame变量名为 `df`。
        请生成一段Python代码来回答问题："{query}"

        要求：
        1. 仅使用pandas操作。
        2. 将最终结果赋值给变量 `result` (字符串或数字)。
        3. 仅输出代码，不要markdown标记。
        """

        try:
            res = self.client.chat.completions.create(
                model=self.config["model_name"],  # ✅ 修正：使用配置模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            code = res.choices[0].message.content
            # 清理 Markdown
            code = code.replace("```python", "").replace("```", "").strip()

            # 沙箱执行
            local_vars = {"df": self.df}
            exec(code, {}, local_vars)

            return str(local_vars.get("result", "无法计算结果"))
        except Exception as e:
            return f"Agent分析失败: {e}"