# 在 retrieve 方法中修改 n_results
    def retrieve(self, query):
        candidates = []

        # 1. Vector Search (扩大到 100)
        if self.encoder:
            query_vec = self.encoder.encode(query).tolist()
            try:
                res = self.collection.query(
                    query_embeddings=[query_vec],
                    n_results=100  # <--- 修改这里：从 30 改到 100
                )
                if res['documents']:
                    candidates.extend(res['documents'][0])
            except:
                pass

        # 2. BM25 Search (扩大到 100)
        if self.bm25:
            tokenized_query = list(jieba.cut(query))
            bm25_docs = self.bm25.get_top_n(
                tokenized_query,
                self.documents,
                n=100  # <--- 修改这里：从 30 改到 100
            )
            candidates.extend(bm25_docs)

        # ... (后续去重和 Rerank 逻辑不变)

        # 去重
        candidates = list(set(candidates))
        if not candidates: return []

        # 3. Rerank (重排序)
        if self.reranker:
            pairs = [[query, doc] for doc in candidates]
            scores = self.reranker.predict(pairs)
            top_indices = np.argsort(scores)[::-1][:5]  # 只取前5
            return [candidates[i] for i in top_indices]

        # 如果没有 Reranker，直接返回前 5
        return candidates[:5]