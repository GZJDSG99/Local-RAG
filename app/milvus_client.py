"""
Milvus 向量数据库模块
=====================

封装与 Milvus 向量数据库的交互，实现：
1. 连接管理
2. 集合（Collection）创建和管理
3. 向量插入
4. 向量检索（相似度搜索）
5. 数据查询和删除

教学要点:
- 向量数据库概念：专门存储和检索高维向量的数据库
- 相似度搜索：使用余弦相似度或欧氏距离度量向量相似性
- 索引类型：HNSW、IVF 等加速大规模向量检索

什么是 Milvus？
- Milvus 是一个开源的向量数据库
- 由 Zilliz 公司开发和维护
- 支持数十亿级向量规模
- 提供多种索引类型和搜索算法
- 可以部署在本地或云端

向量数据库 vs 传统数据库：
┌─────────────────────────────────────────────────────────────┐
│ 传统数据库                │  向量数据库                        │
├─────────────────────────────────────────────────────────────┤
│ 存储结构化数据           │  存储高维向量（嵌入向量）          │
│ 精确匹配查询 (=, LIKE)   │  相似度搜索（最近邻）              │
│ B+树、哈希索引           │  HNSW、IVF、ANNOY 等专用索引       │
│ 适合精确查找             │  适合语义搜索、推荐系统等          │
└─────────────────────────────────────────────────────────────┘

向量检索原理：
1. 用户提问 → 向量化 → 查询向量 Q
2. 在向量空间中找到与 Q 最相似的 K 个向量
3. 返回这些向量对应的原始文本
"""

import sys
from typing import List, Dict, Optional, Any
from pymilvus import MilvusClient, Collection, FieldSchema, CollectionSchema, DataType
from .config import Config


class MilvusDB:
    """
    Milvus 向量数据库客户端

    提供向量存储和检索功能
    """

    def __init__(
        self,
        uri: str = None,
        collection_name: str = None,
        vector_dimension: int = None
    ):
        """
        初始化 Milvus 客户端

        Args:
            uri: Milvus 连接 URI
            collection_name: 集合名称
            vector_dimension: 向量维度
        """
        self.uri = uri or Config.MILVUS_URI
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.vector_dimension = vector_dimension or Config.VECTOR_DIMENSION

        # 创建 Milvus 客户端
        # MilvusClient 是 PyMilvus 2.5.x 推荐的 API
        self._client = MilvusClient(uri=self.uri)

        print(f"[Milvus 客户端] 初始化完成")
        print(f"  连接地址: {self.uri}")
        print(f"  集合名称: {self.collection_name}")
        print(f"  向量维度: {self.vector_dimension}")

    def check_connection(self) -> Dict[str, Any]:
        """
        检查 Milvus 连接状态

        Returns:
            包含连接状态的字典
        """
        try:
            # 尝试获取集合列表来验证连接
            collections = self._client.list_collections()
            return {
                'status': 'connected',
                'uri': self.uri,
                'collections': collections
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def create_collection(self, drop_existing: bool = False) -> bool:
        """
        创建向量集合

        集合（Collection）是 Milvus 中存储向量的容器。
        类似于传统数据库中的"表"概念。

        集合的 Schema 定义：
        - id: 主键，唯一标识每条记录
        - vector: 向量字段，存储文本的嵌入向量
        - text: 原始文本内容
        - metadata: 元数据（文档名、分段索引等）

        Args:
            drop_existing: 如果集合已存在，是否删除后重建

        Returns:
            是否创建成功
        """
        try:
            # 检查集合是否已存在
            if self._client.has_collection(self.collection_name):
                if drop_existing:
                    print(f"[Milvus] 删除已存在的集合: {self.collection_name}")
                    self._client.drop_collection(self.collection_name)
                else:
                    print(f"[Milvus] 集合已存在: {self.collection_name}")
                    return True

            # 创建集合
            # 参数说明：
            # - dimension: 向量维度（必须与 embedding 模型输出一致）
            # - id_type: 主键类型，'int' 表示使用整数主键
            # - metric_type: 距离度量类型
            #   - 'COSINE': 余弦相似度（推荐用于文本嵌入）
            #   - 'L2': 欧氏距离
            #   - 'IP': 内积
            self._client.create_collection(
                collection_name=self.collection_name,
                dimension=self.vector_dimension,
                id_type='int',
                metric_type='COSINE',  # 使用余弦相似度，适合文本向量
                consistency_level=2,  # 最终一致性
            )

            print(f"[Milvus] 集合创建成功: {self.collection_name}")

            # 创建索引以加速搜索
            self._create_index()

            return True

        except Exception as e:
            print(f"[Milvus 错误] 创建集合失败: {str(e)}", file=sys.stderr)
            raise

    def _create_index(self):
        """
        为向量字段创建索引

        索引类型选择指南：
        - HNSW: 精度高，搜索快，但内存占用大。适合小规模数据（百万级）
        - IVF: 精度中等，搜索较快，内存占用中等
        - FLAT: 暴力搜索，无索引，精度最高但最慢

        对于教学场景和小规模数据，使用 FLAT 是最简单可靠的选择。
        """
        try:
            # 使用 FLAT 索引（暴力搜索）- 最简单可靠
            self._client.create_index(
                collection_name=self.collection_name,
                field_name="vector",
                index_type="FLAT",
                index_params={}
            )

            print(f"[Milvus] 索引创建成功: FLAT")

        except Exception as e:
            print(f"[Milvus] 索引已存在或无需创建: {str(e)}")

    def insert_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        插入向量数据

        Args:
            vectors: 向量列表
            texts: 对应的原始文本列表
            metadata: 元数据列表（如文档名、分段索引等）

        Returns:
            插入结果，包含插入的记录数
        """
        if not vectors or not texts:
            raise ValueError("向量和文本列表不能为空")

        if len(vectors) != len(texts):
            raise ValueError("向量和文本列表长度必须一致")

        try:
            # 构建插入数据
            # Milvus 使用字典格式组织数据
            data = []

            for i, (vector, text) in enumerate(zip(vectors, texts)):
                record = {
                    'id': i,  # 主键（递增）
                    'vector': vector,  # 向量
                    'text': text,  # 原始文本
                    'chunk_index': i,  # 分段索引
                    'doc_name': metadata[i].get('doc_name', 'unknown') if metadata else 'unknown',
                    'char_count': len(text)  # 字符数
                }
                data.append(record)

            # 插入数据
            result = self._client.insert(
                collection_name=self.collection_name,
                data=data
            )

            insert_count = result.get('insert_count', len(vectors))
            print(f"[Milvus] 插入成功: {insert_count} 条记录")

            # 自动加载集合到内存，以便立即可以查询
            self._client.load_collection(collection_name=self.collection_name)
            print(f"[Milvus] 集合已加载到内存")

            return {'inserted_count': insert_count}

        except Exception as e:
            print(f"[Milvus 错误] 插入数据失败: {str(e)}", file=sys.stderr)
            raise

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 5,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量

        这是 RAG 流程中的核心步骤：
        1. 用户提问 → 向量化 → 查询向量 Q
        2. 在向量空间中找与 Q 最相似的 K 个向量
        3. 返回这些向量对应的原始文本

        搜索原理：
        - 计算查询向量与所有存储向量的相似度
        - 返回相似度最高的 top-K 个结果
        - 使用 HNSW 索引加速搜索（近似最近邻）

        Args:
            query_vector: 查询向量
            limit: 返回的结果数量
            output_fields: 指定返回哪些字段

        Returns:
            搜索结果列表，每项包含向量数据和相似度分数
        """
        try:
            # 确保集合已加载
            self._client.load_collection(collection_name=self.collection_name)

            if output_fields is None:
                output_fields = ['id', 'text', 'doc_name', 'chunk_index', 'char_count']

            # 执行向量搜索
            # metric_type='COSINE' 需要在 search_params 中指定
            results = self._client.search(
                collection_name=self.collection_name,
                data=[query_vector],  # 搜索向量（必须是列表的列表）
                limit=limit,  # 返回的最相似结果数
                search_params={'metrics': 'COSINE'},  # 指定余弦相似度
                output_fields=output_fields  # 返回的字段
            )

            # 解析结果
            # results 是一个列表的列表，外层列表每个元素对应一个查询向量
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        'id': hit['id'],
                        'text': hit['entity']['text'],
                        'doc_name': hit['entity'].get('doc_name', ''),
                        'chunk_index': hit['entity'].get('chunk_index', 0),
                        'score': hit['distance'],  # 相似度分数
                        'char_count': hit['entity'].get('char_count', 0)
                    })

            print(f"[Milvus] 搜索完成: 找到 {len(search_results)} 条相似结果")

            return search_results

        except Exception as e:
            print(f"[Milvus 错误] 搜索失败: {str(e)}", file=sys.stderr)
            raise

    def query_vectors(
        self,
        filter_expr: str = None,
        limit: int = 100,
        offset: int = 0,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        查询向量数据（标量过滤）

        与 search_vectors 不同：
        - search_vectors: 基于向量相似度搜索
        - query_vectors: 基于字段条件过滤查询

        Args:
            filter_expr: 过滤表达式，如 "doc_name == 'test.md'"
            limit: 返回数量限制
            offset: 跳过前 N 条记录
            output_fields: 返回的字段列表

        Returns:
            查询结果列表
        """
        try:
            if output_fields is None:
                output_fields = ['id', 'text', 'doc_name', 'chunk_index', 'char_count']

            results = self._client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit,
                offset=offset
            )

            print(f"[Milvus] 查询完成: 返回 {len(results)} 条记录")

            return results

        except Exception as e:
            print(f"[Milvus 错误] 查询失败: {str(e)}", file=sys.stderr)
            raise

    def get_all_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        获取集合中的所有数据

        用于查看知识库中已存储的所有分段内容

        Args:
            limit: 最大返回数量

        Returns:
            所有数据记录列表
        """
        try:
            # MilvusClient 没有直接的 "获取所有" 方法
            # 需要通过查询来获取数据
            results = self.query_vectors(limit=limit)

            print(f"[Milvus] 获取所有数据: {len(results)} 条记录")

            return results

        except Exception as e:
            print(f"[Milvus 错误] 获取数据失败: {str(e)}", file=sys.stderr)
            raise

    def get_all_vectors_with_text(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        获取所有存储的向量数据（包括文本）

        用于在构建知识库时同步更新 BM25 索引。

        Args:
            limit: 最大返回数量

        Returns:
            所有数据记录列表，每条记录包含文本和元数据
        """
        try:
            # 确保集合已加载
            self._client.load_collection(collection_name=self.collection_name)

            results = self._client.query(
                collection_name=self.collection_name,
                output_fields=['id', 'text', 'doc_name', 'chunk_index', 'char_count'],
                limit=limit
            )

            # 转换为统一格式
            data = []
            for item in results:
                data.append({
                    'id': item.get('id', 0),
                    'text': item.get('text', ''),
                    'doc_name': item.get('doc_name', ''),
                    'chunk_index': item.get('chunk_index', 0),
                    'char_count': item.get('char_count', 0)
                })

            print(f"[Milvus] 获取所有数据: {len(data)} 条记录")

            return data

        except Exception as e:
            print(f"[Milvus 错误] 获取数据失败: {str(e)}", file=sys.stderr)
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            包含集合信息的字典
        """
        try:
            # 确保集合已加载
            self._client.load_collection(collection_name=self.collection_name)

            stats = self._client.get_collection_stats(
                collection_name=self.collection_name
            )

            return {
                'collection_name': self.collection_name,
                'total_entities': stats.get('total_entities', 0),
                'loaded': stats.get('loaded', False)
            }

        except Exception as e:
            print(f"[Milvus 错误] 获取统计信息失败: {str(e)}", file=sys.stderr)
            raise

    def delete_collection(self) -> bool:
        """
        删除集合

        Returns:
            是否删除成功
        """
        try:
            if self._client.has_collection(self.collection_name):
                self._client.drop_collection(self.collection_name)
                print(f"[Milvus] 集合已删除: {self.collection_name}")
                return True
            else:
                print(f"[Milvus] 集合不存在: {self.collection_name}")
                return False

        except Exception as e:
            print(f"[Milvus 错误] 删除集合失败: {str(e)}", file=sys.stderr)
            raise

    def delete_by_doc_name(self, doc_name: str) -> Dict[str, Any]:
        """
        根据文档名删除数据

        Args:
            doc_name: 文档名称

        Returns:
            删除结果
        """
        try:
            result = self._client.delete(
                collection_name=self.collection_name,
                filter=f"doc_name == '{doc_name}'"
            )

            print(f"[Milvus] 删除文档 '{doc_name}': {result}")

            return result

        except Exception as e:
            print(f"[Milvus 错误] 删除数据失败: {str(e)}", file=sys.stderr)
            raise


# 全局客户端实例
_db_instance: Optional[MilvusDB] = None


def get_milvus_db() -> MilvusDB:
    """
    获取 Milvus 数据库客户端单例

    Returns:
        MilvusDB 实例
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = MilvusDB()
    return _db_instance


def demo_usage():
    """
    使用示例 - 演示 Milvus 客户端的用法
    """
    # 创建客户端
    db = MilvusDB()

    # 1. 检查连接
    print("=" * 50)
    print("1. 检查 Milvus 连接")
    print("=" * 50)
    status = db.check_connection()
    print(f"状态: {status}")

    # 2. 创建集合
    print("\n" + "=" * 50)
    print("2. 创建集合")
    print("=" * 50)
    db.create_collection(drop_existing=True)

    # 3. 插入向量
    print("\n" + "=" * 50)
    print("3. 插入向量数据")
    print("=" * 50)

    # 模拟的向量数据（实际使用时来自 Ollama）
    sample_vectors = [
        [0.1] * 1024,
        [0.2] * 1024,
        [0.3] * 1024,
    ]
    sample_texts = [
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络模型",
        "自然语言处理研究语言理解"
    ]
    sample_metadata = [
        {'doc_name': 'AI简介.txt'},
        {'doc_name': '深度学习.txt'},
        {'doc_name': 'NLP简介.txt'}
    ]

    db.insert_vectors(sample_vectors, sample_texts, sample_metadata)

    # 4. 搜索向量
    print("\n" + "=" * 50)
    print("4. 搜索相似向量")
    print("=" * 50)

    query = [0.15] * 1024  # 查询向量
    results = db.search_vectors(query, limit=2)

    for i, result in enumerate(results):
        print(f"\n结果 {i + 1}:")
        print(f"  文本: {result['text']}")
        print(f"  相似度: {result['score']:.4f}")
        print(f"  文档: {result['doc_name']}")

    # 5. 查询所有数据
    print("\n" + "=" * 50)
    print("5. 查看所有已存储数据")
    print("=" * 50)

    all_data = db.get_all_data()
    print(f"共 {len(all_data)} 条记录")


if __name__ == '__main__':
    demo_usage()
