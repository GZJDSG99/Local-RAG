"""
FAISS 向量数据库模块
=====================

使用 FAISS 实现本地向量存储和检索。

FAISS (Facebook AI Similarity Search) 是一个高效的相似度搜索库，
适合处理百万级向量数据的最近邻搜索。

教学要点:
- 索引类型：IndexFlatL2 (暴力搜索), IndexIVFFlat (倒排索引), IndexHNSW (图索引)
- 相似度搜索：使用欧氏距离或内积度量向量相似性
- 索引构建：需要先训练(Train)再添加向量
"""

import sys
import pickle
import os
from typing import List, Dict, Optional, Any
import numpy as np
import faiss
from .config import Config


class FaissDB:
    """
    FAISS 向量数据库客户端

    提供向量存储和检索功能
    """

    def __init__(
        self,
        index_path: str = None,
        data_path: str = None,
        vector_dimension: int = None
    ):
        """
        初始化 FAISS 客户端

        Args:
            index_path: 索引文件路径
            data_path: 数据文件路径（存储原始文本等）
            vector_dimension: 向量维度
        """
        self.vector_dimension = vector_dimension or Config.VECTOR_DIMENSION
        
        # 使用配置中的数据库路径
        db_dir = os.path.dirname(Config.MILVUS_URI) if Config.MILVUS_URI else './data'
        os.makedirs(db_dir, exist_ok=True)
        
        self.index_path = index_path or os.path.join(db_dir, 'faiss.index')
        self.data_path = data_path or os.path.join(db_dir, 'faiss_data.pkl')
        
        # 索引和数据
        self._index: Optional[faiss.Index] = None
        self._data: List[Dict] = []  # 存储 id, text, doc_name, chunk_index, char_count
        
        # 加载现有索引
        self._load_index()
        
        print(f"[FAISS 客户端] 初始化完成")
        print(f"  索引文件: {self.index_path}")
        print(f"  数据文件: {self.data_path}")
        print(f"  向量维度: {self.vector_dimension}")
        print(f"  当前向量数: {len(self._data)}")

    def _load_index(self):
        """加载现有索引和数据"""
        # 加载索引
        if os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
                print(f"[FAISS] 索引加载成功: {self._index.ntotal} 个向量")
            except Exception as e:
                print(f"[FAISS] 索引加载失败: {e}")
                self._index = None
        
        # 加载数据
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'rb') as f:
                    self._data = pickle.load(f)
                print(f"[FAISS] 数据加载成功: {len(self._data)} 条记录")
            except Exception as e:
                print(f"[FAISS] 数据加载失败: {e}")
                self._data = []

    def _save_index(self):
        """保存索引和数据"""
        try:
            if self._index is not None:
                faiss.write_index(self._index, self.index_path)
                print(f"[FAISS] 索引保存成功: {self.index_path}")
            
            with open(self.data_path, 'wb') as f:
                pickle.dump(self._data, f)
            print(f"[FAISS] 数据保存成功: {self.data_path}")
        except Exception as e:
            print(f"[FAISS] 保存失败: {e}")

    def check_connection(self) -> Dict[str, Any]:
        """
        检查连接状态

        Returns:
            连接状态信息
        """
        return {
            'status': 'connected',
            'vector_count': len(self._data),
            'dimension': self.vector_dimension,
            'index_type': type(self._index).__name__ if self._index else 'None'
        }

    def create_index(self, force: bool = False):
        """
        创建向量索引

        使用 HNSW 索引，适合小规模数据（几千到几万条）
        """
        if self._index is not None and not force:
            print(f"[FAISS] 索引已存在")
            return True
        
        try:
            # 使用 HNSW 索引 - 高召回率，适合教学和小规模数据
            # M: 每个节点的连接数（影响精度和内存）
            # efSearch: 搜索时的搜索范围（影响精度和速度）
            self._index = faiss.IndexHNSWFlat(
                self.vector_dimension,
                32,  # M = 32
                faiss.METRIC_L2
            )
            self._index.hnsw.efSearch = 128  # 搜索参数
            self._index.hnsw.efConstruction = 200  # 构建参数
            
            print(f"[FAISS] HNSW 索引创建成功")
            self._save_index()
            return True
            
        except Exception as e:
            print(f"[FAISS 错误] 创建索引失败: {e}")
            raise

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
            插入结果
        """
        if not vectors or not texts:
            raise ValueError("向量和文本列表不能为空")
        
        if len(vectors) != len(texts):
            raise ValueError("向量和文本列表长度必须一致")
        
        try:
            # 确保索引存在
            if self._index is None:
                self.create_index()
            
            # 转换向量为 numpy 数组
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # 如果维度不匹配，记录警告
            if vectors_array.shape[1] != self.vector_dimension:
                print(f"[FAISS 警告] 向量维度不匹配: 期望 {self.vector_dimension}, 实际 {vectors_array.shape[1]}")
                self.vector_dimension = vectors_array.shape[1]
            
            # 添加到索引
            start_id = len(self._data)
            self._index.add(vectors_array)
            
            # 保存数据
            for i, (text, meta) in enumerate(zip(texts, metadata or [{}] * len(texts))):
                self._data.append({
                    'id': start_id + i,
                    'text': text,
                    'doc_name': meta.get('doc_name', ''),
                    'chunk_index': meta.get('chunk_index', 0),
                    'char_count': len(text)
                })
            
            # 持久化
            self._save_index()
            
            print(f"[FAISS] 插入成功: {len(vectors)} 条向量")
            
            return {
                'status': 'success',
                'inserted_count': len(vectors)
            }
            
        except Exception as e:
            print(f"[FAISS 错误] 插入失败: {e}")
            raise

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 5,
        output_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量

        Args:
            query_vector: 查询向量
            limit: 返回数量
            output_fields: 返回字段（为了兼容性保留此参数）

        Returns:
            搜索结果列表
        """
        if self._index is None or self._index.ntotal == 0:
            print(f"[FAISS] 索引为空，请先添加数据")
            return []
        
        try:
            # 转换查询向量
            query = np.array([query_vector], dtype=np.float32)
            
            # 搜索
            # search 返回距离和索引
            distances, indices = self._index.search(query, limit)
            
            # 解析结果
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:  # 无效索引
                    continue
                
                data = self._data[idx]
                results.append({
                    'id': data['id'],
                    'text': data['text'],
                    'doc_name': data.get('doc_name', ''),
                    'chunk_index': data.get('chunk_index', 0),
                    'score': float(dist),  # L2 距离，越小越相似
                    'char_count': data.get('char_count', 0)
                })
            
            print(f"[FAISS] 搜索完成: 找到 {len(results)} 条相似结果")
            return results
            
        except Exception as e:
            print(f"[FAISS 错误] 搜索失败: {e}")
            raise

    def query_vectors(
        self,
        filter_expr: str = None,
        limit: int = 10,
        output_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查询向量数据

        Args:
            filter_expr: 过滤表达式（为了兼容性保留）
            limit: 返回数量
            output_fields: 返回字段

        Returns:
            查询结果列表
        """
        if not self._data:
            return []
        
        if output_fields is None:
            output_fields = ['id', 'text', 'doc_name', 'chunk_index', 'char_count']
        
        # 返回所有数据（或前 limit 条）
        results = []
        for i, data in enumerate(self._data[:limit]):
            result = {k: data.get(k, '') for k in output_fields if k in data}
            results.append(result)
        
        return results

    def delete_vectors(self, ids: List[int]) -> Dict[str, Any]:
        """
        删除向量（注意：FAISS 不支持高效删除，这里标记但不真正删除）

        Args:
            ids: 要删除的向量 ID 列表

        Returns:
            删除结果
        """
        print(f"[FAISS 警告] FAISS 不支持高效删除向量，数据已标记")
        return {
            'status': 'partial',
            'message': 'FAISS 不支持删除操作，需要重建索引'
        }

    def drop_collection(self):
        """
        删除所有数据
        """
        try:
            self._index = None
            self._data = []
            
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.data_path):
                os.remove(self.data_path)
            
            print(f"[FAISS] 集合已清空")
            return True
            
        except Exception as e:
            print(f"[FAISS 错误] 清空集合失败: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息
        """
        return {
            'vector_count': len(self._data),
            'dimension': self.vector_dimension,
            'index_type': 'HNSW' if isinstance(self._index, faiss.IndexHNSW) else type(self._index).__name__,
            'index_path': self.index_path,
            'data_path': self.data_path
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """兼容 MilvusDB 的方法名"""
        return self.get_stats()
