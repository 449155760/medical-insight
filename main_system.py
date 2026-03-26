import traceback
from openai import OpenAI
import numpy as np

# ⚠️ 注意：不要在文件顶部导入 Core_modules，移到 __init__ 内部
# 防止和其他文件形成导入死循环

class MedicalQASystem:
    def __init__(self, config, chroma_collection, neo4j_driver, docs_text, table_path, ablation_mode="full"):
        self.config = config
        self.client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        self.ablation_mode = ablation_mode

        # ✅ 核心修复：延迟导入 + 正确的路径指向 Core_modules
        # 只有当系统初始化时才加载这些模块
        from Core_modules.factual_module import FactualModule
        from Core_modules.comparison_module import ComparisonModule
        from Core_modules.reasoning_module import ReasoningModule
        from Core_modules.router_module import CascadingRouter

        # 1. 初始化核心组件
        self.factual = FactualModule(config, chroma_collection, docs_text)

        # 2. 共享 Encoder 给 Router 和 Reasoning (节省显存)
        shared_encoder = self.factual.encoder

        self.reasoning = ReasoningModule(config, neo4j_driver, encoder_model=shared_encoder)
        self.comparison = ComparisonModule(config, table_path)
        self.router = CascadingRouter(config, encoder_model=shared_encoder)

    def _get_core_context(self, query):
        """
        【全局重排序策略】收集所有模块的证据，统一打分。
        """
        candidates = []  # list of (content, type)

        # 1. 总是获取事实检索 (Base)
        docs = self.factual.retrieve(query, top_k=5)
        for d in docs: candidates.append((d, "文献检索"))

        # 2. 路由与模块执行
        intent = "factual"

        if self.ablation_mode == "full":
            intent = self.router.route(query)
            if intent == "reasoning":
                chain = self.reasoning.run_tog(query)
                if chain: candidates.append((chain, "图谱推理"))
            elif intent == "comparison":
                table_res = self.comparison.query_table(query)
                if table_res: candidates.append((table_res, "数据分析"))

        elif self.ablation_mode == "no_routing":
            intent = "mixed"
            chain = self.reasoning.run_tog(query)
            if chain: candidates.append((chain, "图谱推理"))
            table_res = self.comparison.query_table(query)
            if table_res: candidates.append((table_res, "数据分析"))

        elif self.ablation_mode == "no_graph":
            intent = self.router.route(query)
            if intent == "comparison":
                table_res = self.comparison.query_table(query)
                if table_res: candidates.append((table_res, "数据分析"))

        elif self.ablation_mode == "no_agent":
            intent = self.router.route(query)
            if intent == "reasoning":
                chain = self.reasoning.run_tog(query)
                if chain: candidates.append((chain, "图谱推理"))

        # 3. 全局去噪 (Global Denoising)
        if len(candidates) <= 3:
            final_docs = candidates
        else:
            # 使用 Factual 的 Reranker 对所有来源的证据统一打分
            texts = [c[0] for c in candidates]
            try:
                pairs = [[query, text] for text in texts]
                scores = self.factual.reranker.predict(pairs)
                sorted_idx = np.argsort(scores)[::-1]
                final_docs = [candidates[i] for i in sorted_idx[:4]]
            except:
                final_docs = candidates[:4]

        # 4. 格式化
        context_str = "\n\n".join([f"【{ctype}】\n{content}" for content, ctype in final_docs])
        return context_str, intent

    def run_for_eval(self, query, options):
        """
        评测专用接口，返回 (Answer, Context, Strategy)
        """
        context, strategy = self._get_core_context(query)
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

        prompt = f"""你是一位资深的临床医生。请根据提供的医学证据，从给出的选项中选出唯一正确答案。

【医学证据】:
{context}

【待评测题目】:
问题: {query}
选项:
{options_text}

【答题规范】:
1. 优先信赖【文献检索】中的定义和数据。
2. 如果【数据分析】的代码结果与常识冲突，可能是代码生成错误，请忽略它。
3. 必须在最后一行给出明确答案，格式为“答案: X”（X为A/B/C/D中的一个）。"""

        try:
            res = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            ans_content = res.choices[0].message.content
            return ans_content, context, strategy
        except Exception as e:
            traceback.print_exc()
            return "ERROR", "NONE", "error"

    def answer_question(self, query):
        """通用问答接口"""
        context, strategy = self._get_core_context(query)
        prompt = f"""基于以下参考信息回答问题。\n{context}\n问题: {query}"""
        try:
            res = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return res.choices[0].message.content
        except:
            return "抱歉，系统暂时无法回答。"