import jieba
import numpy as np
from openai import OpenAI


class ReasoningModule:
    """
    Ours: Vector-guided Chain-of-Thought on Graph (Vector-ToG)
    升级点：使用 LLM 进行精准实体识别 (NER)
    """

    def __init__(self, config, neo4j_driver, encoder_model=None):
        self.config = config
        self.client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        self.driver = neo4j_driver
        self.encoder = encoder_model

    def _extract_entities_llm(self, query):
        """利用 LLM 提取医学实体，不再只取前3个词"""
        prompt = f"""请从以下医疗问题中提取关键的医学实体（症状、疾病、药物、检查项目）。
        仅输出实体，用逗号分隔。不要输出其他内容。
        问题："{query}"
        实体："""
        try:
            resp = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            content = resp.choices[0].message.content.strip()
            # 清洗标点
            keywords = [k.strip() for k in content.replace("，", ",").split(",") if k.strip()]
            return keywords[:5]  # 限制最多5个核心实体
        except:
            # 兜底：如果 LLM 挂了，回退到 Jieba
            return [w for w in jieba.cut(query) if len(w) > 1][:5]

    def run_tog(self, query, top_k=5):
        # 1. 智能实体识别 (NER)
        keywords = self._extract_entities_llm(query)
        # print(f"🔍 [Graph] Extracted Entities: {keywords}") # 调试用

        if not keywords: return ""

        candidates = []

        # 2. 图谱扩展 (Retrieve)
        # 优化 Cypher: 增加 limit，防止漏掉重要邻居
        cypher = """
        MATCH (n)-[r]-(m)
        WHERE n.name CONTAINS $kw 
        RETURN n.name, type(r), m.name
        LIMIT 100 
        """
        # 注意：这里我们只查 n 包含 kw，可以减少一部分噪音，或者保留原来的双向查找

        with self.driver.session() as session:
            for kw in keywords:
                try:
                    res = session.run(cypher, kw=kw)
                    for record in res:
                        path_str = f"{record['n.name']} --[{record['type(r)']}]--> {record['m.name']}"
                        candidates.append(path_str)
                except Exception:
                    pass

        if not candidates: return ""

        # 3. 向量剪枝 (Filter)
        candidates = list(set(candidates))
        try:
            if self.encoder:
                # 批量计算相似度
                path_vecs = self.encoder.encode(candidates)
                query_vec = self.encoder.encode(query)
                scores = np.dot(path_vecs, query_vec)

                # 动态阈值：只保留分数大于 0.4 的路径，或者取 Top K
                top_indices = np.argsort(scores)[::-1][:top_k]

                best_paths = [candidates[i] for i in top_indices]
                return "\n".join(best_paths)
            else:
                return "\n".join(candidates[:top_k])
        except:
            return "\n".join(candidates[:top_k])