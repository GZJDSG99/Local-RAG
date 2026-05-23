"""
BM25 关键词检索模块
==================

基于 Okapi BM25 算法的稀疏检索实现。

BM25 (Best Matching 25) 是一种经典的信息检索算法：
- 考虑词项频率 (TF)
- 考虑逆文档频率 (IDF)
- 考虑文档长度归一化

与向量检索的对比：
┌─────────────────────────────────────────────────────────────────┐
│                  BM25                    │      向量检索           │
├─────────────────────────────────────────────────────────────────┤
│ 稀疏检索（基于词项精确匹配）              │  稠密检索（基于语义理解）  │
│ 擅长：精确关键词、专有名词、专业术语匹配   │  擅长：同义词、语义相似    │
│ 计算快，适合短查询                        │  计算慢，适合长文本理解     │
└─────────────────────────────────────────────────────────────────┘

Hybrid Retrieval = BM25 + 向量检索，结合两者优势提升召回率。
"""

import re
import pickle
import os
from typing import List, Dict, Optional, Any
from rank_bm25 import BM25Okapi
from .config import Config


class BM25Client:
    """
    BM25 关键词检索客户端

    提供文本分词、索引构建和关键词检索功能。
    """

    def __init__(
        self,
        k1: float = None,
        b: float = None
    ):
        """
        初始化 BM25 客户端

        Args:
            k1: BM25 k1 参数，控制词频饱和度
            b: BM25 b 参数，控制文档长度归一化
        """
        self.k1 = k1 or Config.BM25_K1
        self.b = b or Config.BM25_B

        # BM25 模型（延迟初始化）
        self._bm25: Optional[BM25Okapi] = None

        # 存储所有文本，用于返回检索结果
        self._documents: List[str] = []
        self._metadata: List[Dict] = []

        # 中文分词器
        self._tokenizer = self._create_tokenizer()

        print(f"[BM25 客户端] 初始化完成")
        print(f"  k1={self.k1}, b={self.b}")

    def _create_tokenizer(self):
        """
        创建分词器

        使用简单的中文分词策略：
        1. 先用正则分出英文单词和中文
        2. 英文直接转为小写
        3. 中文按字符级别切分（简单但有效的中文处理方式）

        Returns:
            分词函数
        """
        def tokenize(text: str) -> List[str]:
            """
            简单分词器

            Args:
                text: 输入文本

            Returns:
                词列表
            """
            if not text:
                return []

            # 转小写
            text = text.lower()

            # 按非字母/数字/中文字符分割
            # [a-zA-Z0-9\u4e00-\u9fff] 匹配字母、数字、中文
            tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)

            # 过滤太短的词（小于2个字符）
            tokens = [t for t in tokens if len(t) >= 1]

            return tokens

        return tokenize

    def build_index(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """
        构建 BM25 索引

        Args:
            texts: 文本列表
            metadata: 元数据列表
        """
        if not texts:
            print(f"[BM25] 警告：文本列表为空，跳过索引构建")
            return

        print(f"[BM25] 开始构建索引...")

        # 分词
        tokenized_texts = [self._tokenizer(t) for t in texts]

        # 创建 BM25 模型
        self._bm25 = BM25Okapi(
            tokenized_texts,
            k1=self.k1,
            b=self.b
        )

        # 保存原始文本和元数据
        self._documents = texts
        self._metadata = metadata or [{}] * len(texts)

        print(f"[BM25] 索引构建完成")
        print(f"  文档数量: {len(texts)}")
        print(f"  平均文档长度: {sum(len(t) for t in texts) / len(texts):.1f} 字符")

        # 打印 BM25 统计信息
        self._print_stats(tokenized_texts)

    def _print_stats(self, tokenized_texts: List[List[str]]):
        """
        打印 BM25 索引统计信息
        """
        if not tokenized_texts:
            return

        # 词汇表大小
        vocab = set()
        for tokens in tokenized_texts:
            vocab.update(tokens)

        # 每个文档的平均词数
        avg_doc_len = sum(len(tokens) for tokens in tokenized_texts) / len(tokenized_texts)

        # 最常见的词
        from collections import Counter
        all_tokens = [t for tokens in tokenized_texts for t in tokens]
        token_freq = Counter(all_tokens)
        top_tokens = token_freq.most_common(10)

        print(f"  词汇表大小: {len(vocab)}")
        print(f"  平均文档词数: {avg_doc_len:.1f}")
        print(f"  高频词 Top 10: {', '.join(f'{t}({c})' for t, c in top_tokens)}")

    def search(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        BM25 关键词检索

        Args:
            query: 查询文本
            limit: 返回结果数量

        Returns:
            检索结果列表，每项包含文本、分数和元数据
        """
        if self._bm25 is None:
            print(f"[BM25] 警告：索引未构建，返回空结果")
            return []

        if not query:
            return []

        # 分词查询
        query_tokens = self._tokenizer(query)

        if not query_tokens:
            return []

        # 计算每个文档的 BM25 分数
        scores = self._bm25.get_scores(query_tokens)

        # 获取 top-k 结果
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:limit]

        # 构建结果
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的结果
                results.append({
                    'id': idx,
                    'text': self._documents[idx],
                    'score': float(scores[idx]),
                    'metadata': self._metadata[idx],
                    'doc_name': self._metadata[idx].get('doc_name', ''),
                    'chunk_index': self._metadata[idx].get('chunk_index', idx),
                    'retrieval_type': 'bm25'
                })

        return results

    def search_with_scores(
        self,
        query: str,
        limit: int = 5,
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索，返回归一化分数

        Args:
            query: 查询文本
            limit: 返回结果数量
            normalize: 是否对分数进行归一化 (0-1)

        Returns:
            归一化后的检索结果
        """
        results = self.search(query, limit=limit)

        if not results or not normalize:
            return results

        # 获取最大分数用于归一化
        max_score = max(r['score'] for r in results)

        if max_score > 0:
            for r in results:
                r['score'] = r['score'] / max_score

        return results

    def get_document_count(self) -> int:
        """
        获取索引中的文档数量
        """
        return len(self._documents)

    def is_indexed(self) -> bool:
        """
        检查索引是否已构建
        """
        return self._bm25 is not None

    def clear_index(self):
        """
        清空索引
        """
        self._bm25 = None
        self._documents = []
        self._metadata = []
        print(f"[BM25] 索引已清空")

    def save_index(self, path: str):
        """
        保存索引到文件（序列化）

        Args:
            path: 保存路径
        """
        try:
            data = {
                'bm25': self._bm25,
                'documents': self._documents,
                'metadata': self._metadata,
                'k1': self.k1,
                'b': self.b
            }

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            print(f"[BM25] 索引已保存到: {path}")
        except Exception as e:
            print(f"[BM25] 保存索引失败: {str(e)}")

    def load_index(self, path: str):
        """
        从文件加载索引

        Args:
            path: 索引文件路径
        """
        if not os.path.exists(path):
            print(f"[BM25] 索引文件不存在: {path}")
            return

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self._bm25 = data['bm25']
            self._documents = data['documents']
            self._metadata = data['metadata']
            self.k1 = data.get('k1', Config.BM25_K1)
            self.b = data.get('b', Config.BM25_B)

            print(f"[BM25] 索引已加载")
            print(f"  文档数量: {len(self._documents)}")
        except Exception as e:
            print(f"[BM25] 加载索引失败: {str(e)}")


# 全局客户端实例
_bm25_instance: Optional[BM25Client] = None


def get_bm25_client() -> BM25Client:
    """
    获取 BM25 客户端单例

    Returns:
        BM25Client 实例
    """
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Client()
    return _bm25_instance


def demo_usage():
    """
    使用示例 - 演示 BM25 检索的用法
    """
    # 创建客户端
    bm25 = BM25Client()

    # 示例文档
    documents = [
        "机器学习是人工智能的一个重要分支，它使用算法来让计算机从数据中学习。",
        "深度学习是机器学习的一个子领域，使用神经网络模型来处理复杂任务。",
        "自然语言处理研究如何让计算机理解和生成人类语言。",
        "计算机视觉研究如何让计算机理解和分析图像和视频。",
        "强化学习通过试错的方式让智能体学习最优策略。"
    ]

    metadata = [
        {'doc_name': 'ml.txt', 'chunk_index': 0},
        {'doc_name': 'dl.txt', 'chunk_index': 0},
        {'doc_name': 'nlp.txt', 'chunk_index': 0},
        {'doc_name': 'cv.txt', 'chunk_index': 0},
        {'doc_name': 'rl.txt', 'chunk_index': 0}
    ]

    # 构建索引
    print("\n" + "=" * 50)
    print("1. 构建 BM25 索引")
    print("=" * 50)
    bm25.build_index(documents, metadata)

    # 执行检索
    print("\n" + "=" * 50)
    print("2. BM25 检索示例")
    print("=" * 50)

    queries = [
        "机器学习",
        "深度学习 神经网络",
        "计算机 图像"
    ]

    for query in queries:
        print(f"\n查询: '{query}'")
        results = bm25.search(query, limit=3)

        for i, r in enumerate(results):
            print(f"  [{i + 1}] 分数: {r['score']:.4f} - {r['text'][:50]}...")


if __name__ == '__main__':
    demo_usage()
