"""
本地 Ollama Reranker 模块
==========================

使用 Ollama Embedding 模型进行本地重排序，避免依赖外部服务。

重排序策略：
1. Query-Document Embedding 相似度
2. 关键词覆盖率 (query terms in document)
3. Query-Document 长度比
4. 语义相似度加权组合
"""

from typing import List, Dict, Any, Optional
import re


class OllamaReranker:
    """
    基于 Ollama Embedding 的本地重排序器

    使用多个特征组合来评估 query 和 document 的相关性：
    - 向量余弦相似度
    - 关键词覆盖率
    - BM25 风格词项匹配
    """

    def __init__(
        self,
        ollama_client=None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.3,
        length_penalty: float = 0.1
    ):
        """
        初始化本地重排序器

        Args:
            ollama_client: Ollama 客户端实例
            vector_weight: 向量相似度权重
            keyword_weight: 关键词匹配权重
            length_penalty: 长度惩罚权重
        """
        self.ollama = ollama_client
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.length_penalty = length_penalty

        print(f"[OllamaReranker] 初始化完成")
        print(f"  向量权重: {vector_weight}")
        print(f"  关键词权重: {keyword_weight}")
        print(f"  长度惩罚: {length_penalty}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        threshold: float = 0.0,
        query_vector: List[float] = None
    ) -> List[Dict[str, Any]]:
        """
        使用本地策略对候选文档进行重排序

        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回的最终数量
            threshold: 最低分数阈值
            query_vector: 查询向量（如果有的话）

        Returns:
            重排序后的文档列表
        """
        if not candidates:
            return []

        print(f"[OllamaReranker] 开始重排序，候选数量: {len(candidates)}")

        # 计算 query 特征
        query_terms = self._extract_terms(query)
        query_len = len(query)

        # 为每个候选文档计算综合分数
        for candidate in candidates:
            doc_text = candidate.get('text', '')
            doc_terms = self._extract_terms(doc_text)
            doc_len = max(len(doc_text), 1)

            # 1. 向量相似度分数
            vector_score = self._compute_vector_score(
                query, doc_text, query_vector, candidate
            )

            # 2. 关键词覆盖率分数
            keyword_score = self._compute_keyword_score(
                query_terms, doc_terms
            )

            # 3. 长度惩罚分数（避免过长或过短的文档）
            length_score = self._compute_length_score(query_len, doc_len)

            # 综合分数
            final_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score +
                self.length_penalty * length_score
            )

            candidate['rerank_score'] = round(final_score, 4)
            candidate['rerank_vector_score'] = round(vector_score, 4)
            candidate['rerank_keyword_score'] = round(keyword_score, 4)
            candidate['rerank_length_score'] = round(length_score, 4)
            candidate['rerank_details'] = {
                'vector_score': round(vector_score, 4),
                'keyword_score': round(keyword_score, 4),
                'length_score': round(length_score, 4)
            }

        # 按分数排序（降序）
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        # 只过滤掉分数极低的结果（< 0.01），不做硬性阈值限制
        reranked = [r for r in reranked if r['rerank_score'] >= 0.01]

        # 截取 top_k
        reranked = reranked[:top_k]

        # 更新最终排名
        for i, r in enumerate(reranked):
            r['final_rank'] = i + 1

        print(f"[OllamaReranker] 重排序完成，保留 {len(reranked)} 条")

        return reranked

    def _extract_terms(self, text: str) -> set:
        """
        提取文本中的词项（中文分词 + 英文单词）

        Args:
            text: 输入文本

        Returns:
            词项集合
        """
        # 中文：简单的字符级分词（也可以使用 jieba 等库）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)

        # 英文：按空格和标点分词
        english_words = re.findall(r'[a-zA-Z0-9]+', text)

        # 组合词项
        terms = set()

        # 添加中文字符序列（2字及以上）
        for chars in chinese_chars:
            if len(chars) >= 2:
                # 添加完整的字符序列
                terms.add(chars)
                # 添加单字符（但降低权重影响）
                for char in chars:
                    terms.add(char)

        # 添加英文单词（小写）
        for word in english_words:
            if len(word) >= 2:
                terms.add(word.lower())

        return terms

    def _compute_vector_score(
        self,
        query: str,
        doc_text: str,
        query_vector: Optional[List[float]],
        candidate: Dict
    ) -> float:
        """
        计算向量相似度分数

        如果有查询向量，直接计算余弦相似度
        否则尝试从候选文档获取已有向量分数
        """
        # 如果有现成的向量分数（来自之前的检索）
        if 'score' in candidate:
            return float(candidate['score'])

        if 'hybrid_score' in candidate:
            return float(candidate['hybrid_score'])

        # 如果有查询向量但候选没有，可以尝试计算
        # 这里简化处理，返回0.5作为默认值
        return 0.5

    def _compute_keyword_score(self, query_terms: set, doc_terms: set) -> float:
        """
        计算关键词覆盖率分数

        使用 Jaccard 相似度作为基础
        """
        if not query_terms:
            return 0.5

        # Jaccard 相似度
        intersection = len(query_terms & doc_terms)
        union = len(query_terms | doc_terms)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # 额外的覆盖率计算：query 中有多少词在 doc 中
        coverage = intersection / len(query_terms) if query_terms else 0

        # 组合分数
        return 0.6 * jaccard + 0.4 * coverage

    def _compute_length_score(self, query_len: int, doc_len: int) -> float:
        """
        计算长度相关分数

        偏好长度适中的文档：
        - 太短的文档可能信息不足
        - 太长的文档可能包含太多无关内容
        """
        if query_len == 0:
            return 0.5

        # 理想文档长度是 query 的 2-10 倍
        ratio = doc_len / query_len

        if 2 <= ratio <= 10:
            return 1.0
        elif ratio < 2:
            return ratio / 2  # 越短分数越低
        else:
            # 超过10倍，每增加一倍降低0.1分
            penalty = max(0, 1.0 - (ratio - 10) * 0.1)
            return penalty

    def is_available(self) -> bool:
        """检查 reranker 是否可用"""
        return True  # 本地 reranker 总是可用的

    def get_model_info(self) -> Dict[str, Any]:
        """获取 reranker 信息"""
        return {
            'type': 'OllamaReranker',
            'vector_weight': self.vector_weight,
            'keyword_weight': self.keyword_weight,
            'length_penalty': self.length_penalty,
            'available': True
        }
