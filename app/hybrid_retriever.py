"""
Hybrid Retrieval 模块
=====================

结合 BM25（关键词检索）和向量检索的混合检索实现。

Hybrid Retrieval 原理：
┌─────────────────────────────────────────────────────────────────┐
│                         用户查询                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────────┐   ┌──────────┐
        │  BM25    │   │  Vector      │   │  Hybrid │
        │  检索    │   │  检索        │   │  融合   │
        └──────────┘   └──────────────┘   └──────────┘
        (精确词匹配)   (语义相似度)       (RRF/加权)
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   最终排序结果     │
                    │   (高召回率)      │
                    └──────────────────┘

融合策略：
1. RRF (Reciprocal Rank Fusion): 基于排名的融合，简单有效
   score = sum(1 / (k + rank_i)) for each retrieval method

2. 加权分数融合: 基于归一化分数的加权求和
   score = alpha * normalized_vector_score + (1 - alpha) * normalized_bm25_score

3. 纯 RRF: 不依赖分数绝对值，只依赖排名顺序
"""

from typing import List, Dict, Optional, Any, Tuple
from .bm25_client import BM25Client, get_bm25_client
from .config import Config


def get_ollama_reranker(ollama_client=None):
    """获取本地 Ollama Reranker"""
    try:
        from .ollama_reranker import OllamaReranker
        return OllamaReranker(
            ollama_client=ollama_client
        )
    except Exception as e:
        print(f"[OllamaReranker] 加载失败: {e}")
        return None


class HybridRetriever:
    """
    混合检索器

    结合 BM25 关键词检索和向量检索，提供更高召回率的检索能力。
    """

    def __init__(
        self,
        vector_db=None,
        bm25_client: BM25Client = None,
        alpha: float = None,
        rrf_k: int = None,
        bm25_weight: float = None,
        enable_rerank: bool = None,
        ollama_client=None
    ):
        """
        初始化混合检索器

        Args:
            vector_db: 向量数据库实例
            bm25_client: BM25 客户端实例
            alpha: 向量检索权重 (0=纯BM25, 1=纯向量)
            rrf_k: RRF 融合参数
            bm25_weight: BM25 在混合中的权重
            enable_rerank: 是否启用重排序
            ollama_client: Ollama 客户端实例（用于重排序）
        """
        self.vector_db = vector_db
        self.bm25 = bm25_client or get_bm25_client()
        self.ollama_client = ollama_client

        # 配置参数
        self.alpha = alpha if alpha is not None else Config.HYBRID_ALPHA
        self.rrf_k = rrf_k if rrf_k is not None else Config.RRF_K
        self.bm25_weight = bm25_weight if bm25_weight is not None else Config.BM25_WEIGHT
        self.enable_rerank = enable_rerank if enable_rerank is not None else Config.ENABLE_CROSS_ENCODER_RERANK

        # 本地 Ollama Reranker（延迟加载）
        self.reranker = None
        if self.enable_rerank:
            self.reranker = get_ollama_reranker(ollama_client)

        print(f"[Hybrid Retriever] 初始化完成")
        print(f"  向量权重 alpha: {self.alpha}")
        print(f"  RRF k: {self.rrf_k}")
        print(f"  BM25 权重: {self.bm25_weight}")
        print(f"  本地 Reranker: {'启用' if self.reranker else '禁用'}")

    def set_vector_db(self, vector_db):
        """
        设置向量数据库

        Args:
            vector_db: 向量数据库实例
        """
        self.vector_db = vector_db

    def build_index(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """
        构建 BM25 索引

        注意：向量索引在 Milvus 中构建，此处只构建 BM25 索引

        Args:
            texts: 文本列表
            metadata: 元数据列表
        """
        self.bm25.build_index(texts, metadata)

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        vector_limit: int = None,
        bm25_limit: int = None,
        use_rrf: bool = True,
        return_details: bool = False,
        use_rerank: bool = None
    ) -> List[Dict[str, Any]]:
        """
        混合检索

        Args:
            query: 原始查询文本
            query_vector: 查询向量
            top_k: 返回结果数量
            vector_limit: 向量检索返回数量（默认 top_k * 2）
            bm25_limit: BM25 检索返回数量（默认 top_k * 2）
            use_rrf: 是否使用 RRF 融合（否则使用加权分数融合）
            return_details: 是否返回详细分数信息
            use_rerank: 是否使用 CrossEncoder 重排序（默认使用配置值）

        Returns:
            检索结果列表
        """
        vector_limit = vector_limit or (top_k * 2)
        bm25_limit = bm25_limit or (top_k * 2)
        use_rerank = use_rerank if use_rerank is not None else self.enable_rerank

        results = []

        # 1. BM25 检索
        bm25_results = self._search_bm25(query, limit=bm25_limit)

        # 2. 向量检索
        vector_results = self._search_vector(query_vector, limit=vector_limit)

        # 3. 融合结果
        if use_rrf:
            fused_results = self._fusion_rrf(
                bm25_results,
                vector_results,
                top_k=top_k,
                return_details=return_details
            )
        else:
            fused_results = self._fusion_weighted(
                bm25_results,
                vector_results,
                top_k=top_k,
                return_details=return_details
            )

        # 4. CrossEncoder 重排序（如果启用）
        if use_rerank and self.reranker and fused_results:
            fused_results = self._rerank(query, fused_results, top_k=top_k)

        return fused_results

    def search_bm25_only(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        仅使用 BM25 检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            BM25 检索结果
        """
        return self._search_bm25(query, limit=top_k)

    def search_vector_only(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        仅使用向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量

        Returns:
            向量检索结果
        """
        return self._search_vector(query_vector, limit=top_k)

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        query_vector: List[float] = None
    ) -> List[Dict[str, Any]]:
        """
        使用本地 Ollama Reranker 对候选结果进行重排序

        Args:
            query: 查询文本
            candidates: 候选结果列表
            top_k: 返回数量
            query_vector: 查询向量（可选）

        Returns:
            重排序后的结果
        """
        if not self.reranker:
            return candidates

        try:
            # 调用本地 Reranker 重排序
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
                threshold=Config.RERANK_THRESHOLD,
                query_vector=query_vector
            )

            # 添加重排序标记
            for r in reranked:
                r['rerank_applied'] = True
                r['retrieval_type'] = 'hybrid_rrf_rerank'

            return reranked

        except Exception as e:
            print(f"[Hybrid] 本地 Reranker 重排序失败: {e}")
            return candidates

    def _search_bm25(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        执行 BM25 检索

        Args:
            query: 查询文本
            limit: 返回数量

        Returns:
            BM25 检索结果
        """
        if not self.bm25.is_indexed():
            print(f"[Hybrid] BM25 索引未构建，返回空结果")
            return []

        results = self.bm25.search_with_scores(query, limit=limit)

        # 标记检索类型
        for r in results:
            r['retrieval_type'] = 'bm25'

        return results

    def _search_vector(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        执行向量检索

        Args:
            query_vector: 查询向量
            limit: 返回数量

        Returns:
            向量检索结果
        """
        if self.vector_db is None:
            print(f"[Hybrid] 向量数据库未设置，返回空结果")
            return []

        try:
            results = self.vector_db.search_vectors(
                query_vector,
                limit=limit
            )

            # 标记检索类型
            for r in results:
                r['retrieval_type'] = 'vector'

            return results
        except Exception as e:
            print(f"[Hybrid] 向量检索失败: {str(e)}")
            return []

    def _fusion_rrf(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int = 5,
        return_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        使用 RRF (Reciprocal Rank Fusion) 融合结果

        RRF 公式: score = sum(1 / (k + rank_i))

        优点：
        - 不需要分数归一化
        - 对异常分数不敏感
        - 只依赖排名顺序

        Args:
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果
            top_k: 返回数量
            return_details: 是否返回详细分数

        Returns:
            融合后的结果
        """
        # 构建排名字典
        bm25_ranks = {r['id']: r for r in bm25_results}
        vector_ranks = {r['id']: r for r in vector_results}

        # 所有文档 ID
        all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        # 计算 RRF 分数
        fused_scores = {}
        for doc_id in all_ids:
            score = 0.0
            details = {}

            # BM25 排名分数
            if doc_id in bm25_ranks:
                bm25_rank = list(bm25_ranks.keys()).index(doc_id) + 1
                bm25_score = 1.0 / (self.rrf_k + bm25_rank)
                score += bm25_score * self.bm25_weight
                details['bm25_rrf_score'] = bm25_score
                details['bm25_rank'] = bm25_rank
                details['bm25_original_score'] = bm25_ranks[doc_id].get('score', 0)

            # 向量检索排名分数
            if doc_id in vector_ranks:
                vector_rank = list(vector_ranks.keys()).index(doc_id) + 1
                vector_score = 1.0 / (self.rrf_k + vector_rank)
                score += vector_score * (1.0 - self.bm25_weight)
                details['vector_rrf_score'] = vector_score
                details['vector_rank'] = vector_rank
                details['vector_original_score'] = vector_ranks[doc_id].get('score', 0)

            fused_scores[doc_id] = {
                'score': score,
                'details': details,
                'id': doc_id
            }

        # 按分数排序
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]

        # 构建最终结果
        final_results = []
        for i, item in enumerate(sorted_results):
            doc_id = item['id']

            # 合并结果信息
            result = {
                'id': doc_id,
                'rank': i + 1,
                'hybrid_score': item['score'],
                'retrieval_type': 'hybrid_rrf'
            }

            # 添加原始信息
            if doc_id in bm25_ranks:
                result.update({
                    'text': bm25_ranks[doc_id]['text'],
                    'doc_name': bm25_ranks[doc_id].get('doc_name', ''),
                    'chunk_index': bm25_ranks[doc_id].get('chunk_index', doc_id)
                })
            elif doc_id in vector_ranks:
                result.update({
                    'text': vector_ranks[doc_id]['text'],
                    'doc_name': vector_ranks[doc_id].get('doc_name', ''),
                    'chunk_index': vector_ranks[doc_id].get('chunk_index', doc_id)
                })

            if return_details:
                result['fusion_details'] = item['details']

            final_results.append(result)

        return final_results

    def _fusion_weighted(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int = 5,
        return_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        使用加权分数融合结果

        加权公式: score = alpha * norm_vector + (1 - alpha) * norm_bm25

        Args:
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果
            top_k: 返回数量
            return_details: 是否返回详细分数

        Returns:
            融合后的结果
        """
        # 构建查找字典
        bm25_dict = {r['id']: r for r in bm25_results}
        vector_dict = {r['id']: r for r in vector_results}

        # 所有文档 ID
        all_ids = set(bm25_dict.keys()) | set(vector_dict.keys())

        # 归一化分数
        def normalize_scores(results: List[Dict]) -> Dict[int, float]:
            """将分数归一化到 0-1 范围"""
            if not results:
                return {}

            scores = [r['score'] for r in results]
            min_score = min(scores)
            max_score = max(scores)

            if max_score == min_score:
                return {r['id']: 1.0 for r in results}

            normalized = {}
            for r in results:
                normalized[r['id']] = (r['score'] - min_score) / (max_score - min_score)
            return normalized

        bm25_normalized = normalize_scores(bm25_results)
        vector_normalized = normalize_scores(vector_results)

        # 计算加权分数
        fused_scores = {}
        for doc_id in all_ids:
            bm25_score = bm25_normalized.get(doc_id, 0.0)
            vec_score = vector_normalized.get(doc_id, 0.0)

            # 加权融合
            hybrid_score = (
                self.alpha * vec_score +
                (1 - self.alpha) * bm25_score
            )

            details = {
                'bm25_normalized_score': bm25_score,
                'vector_normalized_score': vec_score,
                'bm25_original_score': bm25_dict[doc_id]['score'] if doc_id in bm25_dict else 0,
                'vector_original_score': vector_dict[doc_id]['score'] if doc_id in vector_dict else 0
            }

            fused_scores[doc_id] = {
                'score': hybrid_score,
                'details': details,
                'id': doc_id
            }

        # 按分数排序
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]

        # 构建最终结果
        final_results = []
        for i, item in enumerate(sorted_results):
            doc_id = item['id']

            result = {
                'id': doc_id,
                'rank': i + 1,
                'hybrid_score': item['score'],
                'retrieval_type': 'hybrid_weighted'
            }

            if doc_id in bm25_dict:
                result.update({
                    'text': bm25_dict[doc_id]['text'],
                    'doc_name': bm25_dict[doc_id].get('doc_name', ''),
                    'chunk_index': bm25_dict[doc_id].get('chunk_index', doc_id)
                })
            elif doc_id in vector_dict:
                result.update({
                    'text': vector_dict[doc_id]['text'],
                    'doc_name': vector_dict[doc_id].get('doc_name', ''),
                    'chunk_index': vector_dict[doc_id].get('chunk_index', doc_id)
                })

            if return_details:
                result['fusion_details'] = item['details']

            final_results.append(result)

        return final_results

    def compare_search_methods(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        比较三种检索方法的结果

        用于调试和评估不同检索策略的效果。

        Args:
            query: 查询文本
            query_vector: 查询向量
            top_k: 返回数量

        Returns:
            包含三种检索方法结果的字典
        """
        return {
            'bm25': self.search_bm25_only(query, top_k=top_k),
            'vector': self.search_vector_only(query_vector, top_k=top_k),
            'hybrid_rrf': self.search(
                query,
                query_vector,
                top_k=top_k,
                use_rrf=True
            ),
            'hybrid_weighted': self.search(
                query,
                query_vector,
                top_k=top_k,
                use_rrf=False
            )
        }


# 全局实例
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever(vector_db=None) -> HybridRetriever:
    """
    获取混合检索器单例

    Args:
        vector_db: 可选的向量数据库实例

    Returns:
        HybridRetriever 实例
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(vector_db=vector_db)
    elif vector_db is not None:
        _hybrid_retriever.set_vector_db(vector_db)
    return _hybrid_retriever


def demo_usage():
    """
    使用示例 - 演示混合检索的用法
    """
    from .milvus_client import MilvusDB

    # 创建混合检索器
    hybrid = HybridRetriever()

    # 示例文档
    documents = [
        "机器学习是人工智能的一个重要分支，它使用算法来让计算机从数据中学习。",
        "深度学习是机器学习的一个子领域，使用神经网络模型来处理复杂任务。",
        "自然语言处理研究如何让计算机理解和生成人类语言。",
        "计算机视觉研究如何让计算机理解和分析图像和视频。",
        "强化学习通过试错的方式让智能体学习最优策略。"
    ]

    metadata = [
        {'doc_name': f'doc_{i}.txt', 'chunk_index': i}
        for i in range(len(documents))
    ]

    # 构建 BM25 索引
    print("\n" + "=" * 50)
    print("1. 构建 BM25 索引")
    print("=" * 50)
    hybrid.build_index(documents, metadata)

    # 示例查询
    query = "深度学习 神经网络"
    print(f"\n查询: '{query}'")

    # 比较不同检索方法
    print("\n" + "=" * 50)
    print("2. 比较不同检索方法")
    print("=" * 50)

    # BM25 检索
    print("\n[BM25 检索结果]")
    bm25_results = hybrid.search_bm25_only(query, top_k=3)
    for i, r in enumerate(bm25_results):
        print(f"  {i + 1}. 分数: {r['score']:.4f} - {r['text'][:40]}...")

    # 注意：向量检索需要实际的向量数据库连接
    print("\n[向量检索结果]")
    print("  (需要连接向量数据库)")


if __name__ == '__main__':
    demo_usage()
