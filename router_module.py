import re
from openai import OpenAI
import numpy as np


class CascadingRouter:
    def __init__(self, config, encoder_model=None):
        self.config = config
        self.client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

        # 1. 定义一些“种子问题”用于语义路由 (Semantic Routing)
        self.seed_questions = {
            "comparison": [
                "阿莫西林和头孢的区别是什么？", "这两种药哪个副作用更小？",
                "高血压药的价格对比", "糖尿病和尿崩症的症状差异"
            ],
            "reasoning": [
                "导致心肌梗死的根本原因是什么？", "为什么服用阿司匹林会出血？",
                "这个病的病理生理机制", "诊断的推理过程是怎样的？"
            ]
        }
        self.encoder = encoder_model

        # 预计算种子向量
        if self.encoder:
            self.seed_embeddings = {
                k: self.encoder.encode(v) for k, v in self.seed_questions.items()
            }

    def route(self, query):
        """
        混合路由策略：
        1. 严格正则 (High Precision)
        2. 语义相似度 (High Recall)
        3. LLM 最终裁决 (Context Aware)
        """

        # ---------------------------------------------------------
        # 策略 1: 语义路由 (比正则更智能)
        # ---------------------------------------------------------
        if self.encoder:
            q_vec = self.encoder.encode(query)
            scores = {}
            for intent, seeds in self.seed_embeddings.items():
                # 计算与该意图下所有种子问题的最大相似度
                sims = [np.dot(q_vec, s_vec) for s_vec in seeds]
                scores[intent] = max(sims)

            # 如果与 comparison 的相似度极高 (>0.85)，直接走
            if scores.get("comparison", 0) > 0.85:
                return "comparison", scores["comparison"]

        # ---------------------------------------------------------
        # 策略 2: LLM Few-Shot (更懂上下文)
        # ---------------------------------------------------------
        return self._llm_route_few_shot(query)

    def _llm_route_few_shot(self, query):
        # 增加 Few-Shot 示例，教 LLM 怎么选
        prompt = f"""你是一个医疗查询路由器。请将用户问题分类为以下之一：
        - comparison: 需要对比多个实体的属性（如价格、副作用、区别）。通常需要查表格。
        - reasoning: 需要推导因果、诊断、机制。通常需要查图谱。
        - factual: 简单的定义或事实查询。查文档即可。

        示例 1:
        Q: "阿司匹林和布洛芬哪个更伤胃？"
        A: comparison

        示例 2:
        Q: "阿司匹林导致胃出血的机制是什么？"
        A: reasoning (注意：虽然有阿司匹林，但问的是机制，不是对比)

        示例 3:
        Q: "什么是吉兰巴雷综合征？"
        A: factual

        用户问题: "{query}"
        分类结果:"""

        try:
            res = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            intent = res.choices[0].message.content.strip().lower()

            # 清洗输出
            if "comparison" in intent: return "comparison", 0.9
            if "reasoning" in intent: return "reasoning", 0.9
            return "factual", 0.9
        except:
            return "factual", 0.5