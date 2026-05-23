"""
CrossEncoder Reranker 模块
===========================

CrossEncoder 是一种高效的重排序模型，它可以同时接收 query 和 document，
计算精确的相关性分数，显著提升检索结果的排序质量。

工作原理：
1. BM25 + Vector 检索返回候选文档列表
2. CrossEncoder 对每个 (query, document) 对进行打分
3. 根据 CrossEncoder 分数重新排序，保留最相关的文档

优势：
- 比单独使用 embedding 的余弦相似度更准确
- 可以捕获 query 和 document 之间的复杂交互
- 适合作为第二阶段的重排序
"""

from typing import List, Dict, Any, Optional
import numpy as np


class CrossEncoderReranker:
    """
    CrossEncoder 重排序器

    使用 sentence-transformers 的 CrossEncoder 模型进行重排序
    """

    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-base',
        device: str = 'cpu',
        max_length: int = 512
    ):
        """
        初始化 CrossEncoder 重排序器

        Args:
            model_name: 模型名称，支持 HuggingFace 上的所有 CrossEncoder 模型
            device: 运行设备 ('cpu', 'cuda', 'mps')
            max_length: 最大输入长度
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化模型，只在第一次使用时加载"""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder
            import torch

            print(f"[CrossEncoder] 正在加载模型: {self.model_name}")

            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )

            self._initialized = True
            print(f"[CrossEncoder] 模型加载成功")

        except ImportError as e:
            print(f"[CrossEncoder] 缺少依赖库: {e}")
            print("[CrossEncoder] 请运行: pip install sentence-transformers torch")
            self._initialized = False
        except Exception as e:
            print(f"[CrossEncoder] 模型加载失败: {e}")
            self._initialized = False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        使用 CrossEncoder 对候选文档进行重排序

        Args:
            query: 查询文本
            candidates: 候选文档列表，每个文档包含 'id', 'text' 等字段
            top_k: 返回的最终数量
            threshold: 最低分数阈值，低于此分数的文档将被过滤

        Returns:
            重排序后的文档列表，包含 'rerank_score' 字段
        """
        if not candidates:
            return []

        # 延迟初始化
        self._lazy_init()

        if not self._initialized or self.model is None:
            print("[CrossEncoder] 模型未加载，跳过重排序")
            return candidates

        try:
            # 准备输入对
            sentence_pairs = [
                (query, candidate.get('text', '')) for candidate in candidates
            ]

            # 计算相关性分数
            scores = self.model.predict(
                sentence_pairs,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # 如果是单个分数（标量），转换为数组
            if isinstance(scores, np.floating):
                scores = np.array([scores])
            elif not isinstance(scores, np.ndarray):
                scores = np.array(scores)

            # 将分数添加到候选文档
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(scores[i])
                candidate['original_rank'] = i + 1

            # 按分数排序（降序）
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

            # 应用阈值过滤
            reranked = [r for r in reranked if r['rerank_score'] >= threshold]

            # 截取 top_k
            reranked = reranked[:top_k]

            # 更新最终排名
            for i, r in enumerate(reranked):
                r['final_rank'] = i + 1

            print(f"[CrossEncoder] 重排序完成，从 {len(candidates)} 条候选中保留 {len(reranked)} 条")
            return reranked

        except Exception as e:
            print(f"[CrossEncoder] 重排序失败: {e}")
            return candidates

    def compute_similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """
        计算 query 与多个文档的相关性分数

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            相关性分数数组
        """
        if not documents:
            return np.array([])

        self._lazy_init()

        if not self._initialized or self.model is None:
            return np.zeros(len(documents))

        try:
            sentence_pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(
                sentence_pairs,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            if isinstance(scores, np.floating):
                return np.array([float(scores)])
            return scores

        except Exception as e:
            print(f"[CrossEncoder] 相似度计算失败: {e}")
            return np.zeros(len(documents))

    def is_available(self) -> bool:
        """检查 CrossEncoder 是否可用"""
        self._lazy_init()
        return self._initialized and self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'initialized': self._initialized,
            'available': self.is_available()
        }
