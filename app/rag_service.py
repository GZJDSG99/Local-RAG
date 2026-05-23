"""
RAG 服务模块
============

RAG (Retrieval-Augmented Generation) 核心实现

整合文档解析、文本分段、向量化、向量存储和问答生成，
提供完整的 RAG 问答流程。

教学要点:
- RAG 流程：检索 + 生成
- 端到端流程：从文档上传到获取回答
- 各环节的技术原理和参数调优

RAG 完整流程图：
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG 问答流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【阶段1：知识库构建】                                                       │
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │ 上传文档  │───→│ 文档解析器   │───→│ 文本分段器  │───→│ Ollama Embed  │  │
│  │ .md/.txt │    │ 提取文本内容 │    │ 切成小段落  │    │ 转换为向量    │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────────────┘  │
│                                                                │            │
│                                                                ↓            │
│                                                         ┌───────────────┐  │
│                                                         │ Milvus 数据库  │  │
│                                                         │ 存储向量数据   │  │
│                                                         └───────────────┘  │
│                                                                             │
│  【阶段2：问答流程】                                                         │
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │ 用户提问  │───→│ Ollama Embed │───→│ Milvus 检索  │───→│ 上下文构建    │  │
│  │ "如何..." │    │ 转换为向量    │    │ 找相似段落   │    │ 拼接相关片段  │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────────────┘  │
│                                                                    │       │
│                                                                    ↓       │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────────┐       │
│  │ 返回回答  │←───│ Ollama Chat │←───│ LLM 基于上下文生成答案        │       │
│  │ 给用户   │    │ 生成回答     │    │ 结合知识 + 模型能力            │       │
│  └──────────┘    └──────────────┘    └─────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import sys
from typing import List, Dict, Optional, Any
from werkzeug.datastructures import FileStorage

from .document_parser import DocumentParser
from .text_chunker import TextChunker
from .ollama_client import OllamaClient, get_ollama_client
from .faiss_client import FaissDB
from .milvus_client import MilvusDB, get_milvus_db
from .bm25_client import BM25Client, get_bm25_client
from .hybrid_retriever import HybridRetriever, get_hybrid_retriever
from .config import Config


# 设置为 True 启用 FAISS，False 启用 Milvus
USE_FAISS = False

# 设置为 True 启用 Hybrid Retrieval (BM25 + Vector)
USE_HYBRID_RETRIEVAL = True


class RAGService:
    """
    RAG 服务类

    提供完整的知识库问答功能：
    - build_knowledge_base(): 从文档构建知识库
    - query(): 基于知识库的问答
    - get_knowledge_base(): 查看知识库内容
    """

    def __init__(
        self,
        ollama_client: OllamaClient = None,
        vector_db=None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        初始化 RAG 服务

        Args:
            ollama_client: Ollama 客户端实例
            vector_db: 向量数据库实例（默认根据 USE_FAISS 选择）
            chunk_size: 文本分段大小
            chunk_overlap: 分段重叠大小
        """
        self.ollama = ollama_client or get_ollama_client()

        # 根据配置选择向量数据库
        if vector_db is not None:
            self.vector_db = vector_db
        elif USE_FAISS:
            self.vector_db = get_faiss_db()
        else:
            self.vector_db = get_milvus_db()
            # 确保 Milvus 集合已创建
            try:
                self.vector_db.create_collection(drop_existing=False)
            except:
                pass

        self.chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP

        # 初始化文本分段器
        self.chunker = TextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=Config.DEFAULT_SEPARATORS
        )

        # 初始化 Hybrid Retriever
        if USE_HYBRID_RETRIEVAL:
            self.hybrid_retriever = get_hybrid_retriever(self.vector_db)
            # 尝试加载现有数据到 BM25 索引
            self._load_bm25_index()
        else:
            self.hybrid_retriever = None

        # 确保索引已创建
        self._ensure_index()

        db_type = "FAISS" if USE_FAISS else "Milvus"
        retrieval_type = "Hybrid (BM25 + Vector)" if USE_HYBRID_RETRIEVAL else "Vector Only"
        print(f"[RAG 服务] 初始化完成 (DB: {db_type}, Retrieval: {retrieval_type})")
        print(f"  分段大小: {self.chunk_size} 字符")
        print(f"  分段重叠: {self.chunk_overlap} 字符")

    def _ensure_index(self):
        """
        确保向量索引存在
        """
        try:
            # 尝试调用 check_connection 来验证连接
            self.vector_db.check_connection()
        except Exception as e:
            print(f"[RAG 服务] 向量数据库连接失败: {str(e)}")
            # 对于 FAISS，尝试创建索引
            if hasattr(self.vector_db, '_index'):
                try:
                    if self.vector_db._index is None:
                        self.vector_db.create_index()
                except:
                    self.vector_db.create_index()

    def _load_bm25_index(self):
        """
        加载现有数据到 BM25 索引

        当启用 Hybrid Retrieval 时，需要将 Milvus 中已有的数据加载到 BM25 索引
        """
        try:
            # 获取所有现有数据
            all_data = self.vector_db.get_all_vectors_with_text(limit=10000)

            if not all_data:
                print(f"[RAG 服务] 知识库为空，跳过 BM25 索引加载")
                return

            # 提取文本和元数据
            texts = [item['text'] for item in all_data]
            metadata = [
                {
                    'doc_name': item.get('doc_name', ''),
                    'chunk_index': item.get('chunk_index', i)
                }
                for i, item in enumerate(all_data)
            ]

            # 构建 BM25 索引
            self.hybrid_retriever.build_index(texts, metadata)
            print(f"[RAG 服务] BM25 索引已加载，共 {len(texts)} 条记录")

        except Exception as e:
            print(f"[RAG 服务] 加载 BM25 索引失败: {str(e)}")

    def build_knowledge_base(
        self,
        file: FileStorage,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ) -> Dict[str, Any]:
        """
        从文档构建知识库

        这是知识库构建的完整流程：
        1. 解析文档 → 提取文本内容
        2. 文本分段 → 切分成小块
        3. 向量化 → 调用 Ollama Embedding API
        4. 存储 → 保存到 Milvus

        Args:
            file: 上传的文件对象
            chunk_size: 分段大小
            chunk_overlap: 分段重叠
            separators: 分隔符列表

        Returns:
            构建结果统计
        """
        print(f"\n{'=' * 60}")
        print(f"开始构建知识库: {file.filename}")
        print(f"{'=' * 60}\n")

        # 记录开始时间
        import time
        start_time = time.time()

        # ========== 步骤 1: 解析文档 ==========
        print("[步骤 1/4] 解析文档...")
        try:
            content = DocumentParser.parse_document(file)
            print(f"  [OK] 文档解析完成，字符数: {len(content)}")
        except Exception as e:
            print(f"  [FAIL] 文档解析失败: {str(e)}")
            raise

        # ========== 步骤 2: 文本分段 ==========
        print("\n[步骤 2/4] 文本分段...")

        # 动态更新分段参数
        if chunk_size is not None:
            self.chunker.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunker.chunk_overlap = chunk_overlap
        if separators:
            self.chunker.separators = separators

        # 分段
        chunks = self.chunker.chunk_text(content)

        print(f"  [OK] 分段完成，共 {len(chunks)} 个分段")
        print(f"  分段参数: size={self.chunker.chunk_size}, overlap={self.chunker.chunk_overlap}")
        if separators:
            print(f"  分隔符: {self.chunker.separators}")

        if chunks:
            print(f"  示例分段 (前200字符): {chunks[0][:200]}...")

        # ========== 步骤 3: 向量化 ==========
        print("\n[步骤 3/4] 生成向量嵌入...")

        # 构建元数据列表
        metadata = [
            {
                'doc_name': file.filename,
                'chunk_index': i
            }
            for i in range(len(chunks))
        ]

        try:
            vectors = self.ollama.embed_texts(chunks)
            print(f"  [OK] 向量化完成，共 {len(vectors)} 个向量")
            print(f"  向量维度: {len(vectors[0]) if vectors else 0}")
        except Exception as e:
            print(f"  [FAIL] 向量化失败: {str(e)}")
            raise

        # ========== 步骤 4: 存储到 Milvus ==========
        print("\n[步骤 4/4] 存储到向量数据库...")

        try:
            result = self.vector_db.insert_vectors(vectors, chunks, metadata)
            print(f"  [OK] 存储完成，插入 {result['inserted_count']} 条记录")
        except Exception as e:
            print(f"  [FAIL] 存储失败: {str(e)}")
            raise

        # ========== 步骤 5: 更新 BM25 索引 ==========
        if USE_HYBRID_RETRIEVAL and self.hybrid_retriever:
            print("\n[步骤 5/5] 更新 BM25 索引...")

            try:
                # 获取 Milvus 中的所有数据（原有数据 + 新数据）
                all_data = self.vector_db.get_all_vectors_with_text(limit=10000)

                if all_data:
                    all_texts = [item['text'] for item in all_data]
                    all_metadata = [
                        {
                            'doc_name': item.get('doc_name', ''),
                            'chunk_index': item.get('chunk_index', i)
                        }
                        for i, item in enumerate(all_data)
                    ]
                    # 构建完整的 BM25 索引（包含所有数据）
                    self.hybrid_retriever.build_index(all_texts, all_metadata)
                    print(f"  [OK] BM25 索引更新完成，共 {len(all_texts)} 条记录")
                else:
                    # 极端情况：Milvus 中没有数据
                    self.hybrid_retriever.build_index(chunks, metadata)
                    print(f"  [OK] BM25 索引更新完成，共 {len(chunks)} 条记录")
            except Exception as e:
                print(f"  [!] BM25 索引更新失败: {str(e)}，不影响检索功能")

        # 计算耗时
        elapsed_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"知识库构建完成!")
        print(f"  文档: {file.filename}")
        print(f"  分段数: {len(chunks)}")
        print(f"  耗时: {elapsed_time:.2f} 秒")
        print(f"{'=' * 60}\n")

        return {
            'status': 'success',
            'filename': file.filename,
            'total_chars': len(content),
            'chunk_count': len(chunks),
            'elapsed_time': elapsed_time
        }

    def query(
        self,
        question: str,
        top_k: int = None,
        similarity_threshold: float = None,
        history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        基于知识库的问答

        问答流程：
        0. 意图识别 → 判断问题是否需要知识库检索
        1. 向量化 → 将用户问题转换为向量
        2. 相似度检索 → 在 Milvus 中找相似分段
        3. 上下文构建 → 拼接相关分段作为上下文
        4. LLM 生成 → 调用 Chat API 生成回答

        Args:
            question: 用户问题
            top_k: 检索的相关分段数量
            similarity_threshold: 相似度阈值
            history: 对话历史列表 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            包含回答和相关分段的字典
        """
        history = history or []  # 默认空列表

        print(f"\n{'=' * 60}")
        print(f"开始处理问答...")
        print(f"问题: {question}")
        if history:
            print(f"对话历史: {len(history)} 条消息 ({len(history) // 2} 轮)")
        print(f"{'=' * 60}\n")

        import time
        start_time = time.time()

        top_k = top_k or Config.TOP_K
        similarity_threshold = similarity_threshold or Config.SIMILARITY_THRESHOLD

        # ========== 步骤 0: 意图识别 ==========
        print("[步骤 0/5] 意图识别...")

        try:
            intent_result = self.ollama.classify_intent(question)
            intent = intent_result.get('intent', 'KNOWLEDGE_QA')
            confidence = intent_result.get('confidence', 0.5)
            reasoning = intent_result.get('reasoning', '')
            need_retrieval = intent_result.get('need_retrieval', True)
            suggested_rewrite = intent_result.get('suggested_rewrite', question)

            print(f"  意图: {intent} (置信度: {confidence:.2f})")
            print(f"  依据: {reasoning}")

            # 如果置信度过低，默认走检索
            if confidence < Config.INTENT_CONFIDENCE_THRESHOLD:
                print(f"  [!] 置信度低于阈值({Config.INTENT_CONFIDENCE_THRESHOLD})，默认走检索")
                need_retrieval = True

        except Exception as e:
            print(f"  [FAIL] 意图识别失败: {str(e)}，默认走检索")
            intent = 'KNOWLEDGE_QA'
            need_retrieval = True
            confidence = 0.5

        # ========== 处理不同意图 ==========
        # 闲聊类：直接回复
        if intent == 'CHITCHAT':
            print(f"\n[意图识别] 闲聊类问题，直接回复")
            elapsed_time = time.time() - start_time

            return {
                'status': 'success',
                'question': question,
                'answer': '你好！有什么我可以帮助你的吗？如果你有任何问题需要从知识库中查找答案，请随时告诉我。',
                'relevant_chunks': [],
                'intent': intent,
                'elapsed_time': elapsed_time
            }

        # 超出范围：拒绝回答
        if intent == 'OUT_OF_SCOPE':
            print(f"\n[意图识别] 超出范围的问题，拒绝回答")
            elapsed_time = time.time() - start_time

            return {
                'status': 'success',
                'question': question,
                'answer': '抱歉，这个问题超出了我能够回答的范围。请尝试询问与知识库内容相关的问题。',
                'relevant_chunks': [],
                'intent': intent,
                'elapsed_time': elapsed_time
            }

        # 系统命令：特殊处理（这里简化为提示）
        if intent == 'SYSTEM_CMD':
            print(f"\n[意图识别] 系统命令，提示用户")
            elapsed_time = time.time() - start_time

            return {
                'status': 'success',
                'question': question,
                'answer': f'我收到了你的指令。但目前系统暂不支持该命令。请尝试提问知识库相关问题。',
                'relevant_chunks': [],
                'intent': intent,
                'elapsed_time': elapsed_time
            }

        # 后续问题：使用改写后的问题
        if intent == 'FOLLOWUP_QUERY' and suggested_rewrite:
            print(f"\n[意图识别] 后续问题，改写为: {suggested_rewrite}")
            question = suggested_rewrite

        # ========== 步骤 1: 问题向量化 ==========
        print("[步骤 1/5] 将问题转换为向量...")

        try:
            query_vector = self.ollama.embed_text(question)
            print(f"  [OK] 向量化完成，向量维度: {len(query_vector)}")
        except Exception as e:
            print(f"  [FAIL] 问题向量化失败: {str(e)}")
            raise

        # ========== 步骤 2: 相似度检索 ==========
        retrieval_type = "Hybrid (BM25 + Vector)" if USE_HYBRID_RETRIEVAL else "Vector Only"
        print(f"\n[步骤 2/5] 在知识库中检索相关分段 (使用 {retrieval_type})...")

        try:
            if USE_HYBRID_RETRIEVAL and self.hybrid_retriever:
                # Hybrid Retrieval: BM25 + Vector Search
                search_results = self.hybrid_retriever.search(
                    query=question,
                    query_vector=query_vector,
                    top_k=top_k * 2,
                    use_rrf=True,
                    return_details=True
                )
                print(f"  [OK] Hybrid 检索完成，初步找到 {len(search_results)} 条结果")

                # 统计各类型检索结果
                bm25_count = sum(1 for r in search_results if r.get('retrieval_type') == 'hybrid_rrf')
                print(f"  Hybrid 融合结果: {len(search_results)} 条")

                # 过滤低相似度结果（使用 hybrid_score）
                relevant_chunks = [
                    r for r in search_results
                    if r.get('hybrid_score', 0) > 0
                ][:top_k]
            else:
                # 纯向量检索（向后兼容）
                search_results = self.vector_db.search_vectors(
                    query_vector,
                    limit=top_k * 2
                )
                print(f"  [OK] 向量检索完成，初步找到 {len(search_results)} 条结果")

                relevant_chunks = [
                    r for r in search_results
                    if r.get('hybrid_score', r.get('score', 0)) >= similarity_threshold
                ][:top_k]

        except Exception as e:
            print(f"  [FAIL] 检索失败: {str(e)}")
            raise

        print(f"  过滤后保留 {len(relevant_chunks)} 条")

        if not relevant_chunks:
            print(f"  [!] 未找到足够相似的相关分段，知识库可能为空或内容不相关")

        # 打印检索到的分段
        for i, chunk in enumerate(relevant_chunks):
            score = chunk.get('hybrid_score', chunk.get('score', 0))
            retrieval_info = chunk.get('retrieval_type', 'vector')
            print(f"\n  [相关分段 {i + 1}] 分数: {score:.4f} ({retrieval_info})")
            print(f"  来源: {chunk['doc_name']} (分段 {chunk['chunk_index']})")
            print(f"  内容: {chunk['text'][:150]}...")

        # ========== 步骤 3: 构建上下文 ==========
        print("\n[步骤 3/5] 构建 Prompt...")

        # 拼接相关分段作为上下文
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"【文档 {i + 1}】(来源: {chunk['doc_name']}, 相似度: {chunk.get('hybrid_score', chunk.get('score', 0)):.2f})\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        # 构建完整的提示词
        user_prompt = Config.USER_PROMPT_TEMPLATE.format(
            context=context if context else "（知识库中未找到相关内容）",
            question=question
        )

        print(f"  [OK] Prompt 构建完成")
        print(f"  上下文长度: {len(context)} 字符")

        # ========== 步骤 4: LLM 生成回答 ==========
        print("\n[步骤 4/5] 生成回答...")

        try:
            # 构建消息列表，包含历史记录
            messages = []

            # 添加历史对话
            for msg in history:
                if msg.get('role') == 'user':
                    messages.append({'role': 'user', 'content': msg.get('content', '')})
                elif msg.get('role') == 'assistant':
                    messages.append({'role': 'assistant', 'content': msg.get('content', '')})

            # 添加当前用户消息（包含上下文的 Prompt）
            messages.append({'role': 'user', 'content': user_prompt})

            print(f"  对话消息数量: {len(messages)}, 历史轮次: {len(history) // 2}")

            answer = self.ollama.chat(
                messages=messages,
                system_prompt=Config.SYSTEM_PROMPT,
                temperature=0.7
            )

            print(f"  [OK] 回答生成完成，长度: {len(answer)} 字符")

        except Exception as e:
            print(f"  [FAIL] 回答生成失败: {str(e)}")
            raise

        # 计算耗时
        elapsed_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"问答处理完成!")
        print(f"  耗时: {elapsed_time:.2f} 秒")
        print(f"{'=' * 60}\n")

        return {
            'status': 'success',
            'question': question,
            'answer': answer,
            'relevant_chunks': relevant_chunks,
            'intent': intent,
            'elapsed_time': elapsed_time
        }

    def get_knowledge_base(
        self,
        doc_name: str = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        获取知识库中的数据

        Args:
            doc_name: 按文档名筛选（可选）
            limit: 返回数量限制

        Returns:
            知识库数据统计和内容
        """
        try:
            # 获取集合统计
            stats = self.vector_db.get_collection_stats()

            # 获取数据
            if doc_name:
                filter_expr = f"doc_name == '{doc_name}'"
                data = self.vector_db.query_vectors(
                    filter_expr=filter_expr,
                    limit=limit
                )
            else:
                data = self.vector_db.query_vectors(limit=limit)

            # 按文档分组
            docs_map = {}
            for item in data:
                doc = item.get('doc_name', 'unknown')
                if doc not in docs_map:
                    docs_map[doc] = []
                docs_map[doc].append(item)

            return {
                'status': 'success',
                'total_entities': stats.get('vector_count', len(data)),
                'documents': list(docs_map.keys()),
                'document_count': len(docs_map),
                'data': data,
                'by_document': docs_map
            }

        except Exception as e:
            print(f"[RAG 服务] 获取知识库失败: {str(e)}", file=sys.stderr)
            raise

    def clear_knowledge_base(self, doc_name: str = None) -> Dict[str, Any]:
        """
        清除知识库

        Args:
            doc_name: 如果指定，只删除该文档；否则删除整个知识库

        Returns:
            操作结果
        """
        try:
            if doc_name:
                # 只删除指定文档
                result = self.vector_db.delete_by_doc_name(doc_name)

                # 重建 BM25 索引（删除指定文档后的所有数据）
                if USE_HYBRID_RETRIEVAL and self.hybrid_retriever:
                    self._load_bm25_index()

                return {
                    'status': 'success',
                    'message': f'已删除文档: {doc_name}',
                    'doc_name': doc_name
                }
            else:
                # 删除整个集合并重建
                self.vector_db.delete_collection()
                self.vector_db.create_collection(drop_existing=False)

                # 清空 BM25 索引
                if USE_HYBRID_RETRIEVAL and self.hybrid_retriever:
                    self.hybrid_retriever.bm25.clear_index()

                return {
                    'status': 'success',
                    'message': '已清空整个知识库'
                }
        except Exception as e:
            print(f"[RAG 服务] 清除知识库失败: {str(e)}", file=sys.stderr)
            raise


# 全局服务实例
_rag_service: Optional[RAGService] = None
_vector_db: Optional[FaissDB] = None


def get_faiss_db() -> FaissDB:
    """
    获取 FAISS 数据库单例

    Returns:
        FaissDB 实例
    """
    global _vector_db
    if _vector_db is None:
        _vector_db = FaissDB()
    return _vector_db


def get_rag_service() -> RAGService:
    """
    获取 RAG 服务单例

    Returns:
        RAGService 实例
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def demo_usage():
    """
    使用示例 - 演示 RAG 服务的完整用法
    """
    # 创建 RAG 服务
    rag = RAGService()

    # 1. 检查系统状态
    print("\n" + "=" * 60)
    print("1. 检查系统连接状态")
    print("=" * 60)

    ollama_status = rag.ollama.check_connection()
    vector_db_status = rag.vector_db.check_connection()

    print(f"Ollama 状态: {ollama_status['status']}")
    if ollama_status['status'] == 'connected':
        print(f"  可用模型: {', '.join(ollama_status.get('available_models', []))}")

    print(f"向量数据库状态: {vector_db_status['status']}")
    print(f"  向量数量: {vector_db_status.get('vector_count', 0)}")

    # 2. 问答示例
    print("\n" + "=" * 60)
    print("2. 问答示例")
    print("=" * 60)

    question = "什么是机器学习？"

    try:
        result = rag.query(question)
        print(f"\n问题: {question}")
        print(f"\n回答:\n{result['answer']}")
        print(f"\n相关分段数: {len(result['relevant_chunks'])}")
    except Exception as e:
        print(f"问答失败: {str(e)}")


if __name__ == '__main__':
    demo_usage()
