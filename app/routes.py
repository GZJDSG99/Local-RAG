"""
Flask 路由模块
==============

定义所有 API 路由和视图函数。

教学要点:
- RESTful API 设计
- Flask 路由和请求处理
- JSON 响应格式
- 文件上传处理
- 错误处理和日志
"""

import os
from flask import Blueprint, request, jsonify, render_template, Response
import json

# 导入服务
from .rag_service import get_rag_service, USE_HYBRID_RETRIEVAL
from .ollama_client import get_ollama_client
from .faiss_client import FaissDB
from .config import Config
from .conversation_service import get_conversation_service

# 创建 Blueprint
api_bp = Blueprint('api', __name__)

# ============ 根路由 - 返回前端页面 ============

@api_bp.route('/')
def index():
    """
    返回前端页面
    """
    return render_template('index.html')


# ============ 系统状态 API ============

@api_bp.route('/api/status')
def get_status():
    """
    获取系统状态

    返回 Ollama 和向量数据库的连接状态
    """
    try:
        rag = get_rag_service()

        # 检查 Ollama 连接
        ollama_status = rag.ollama.check_connection()

        # 检查向量数据库连接
        vector_db_status = rag.vector_db.check_connection()

        # 获取知识库统计
        try:
            kb_stats = rag.get_knowledge_base()
            kb_info = {
                'total_entities': kb_stats.get('total_entities', 0),
                'document_count': kb_stats.get('document_count', 0),
                'documents': kb_stats.get('documents', [])
            }
        except:
            kb_info = {
                'total_entities': 0,
                'document_count': 0,
                'documents': []
            }

        return jsonify({
            'status': 'success',
            'ollama': {
                'status': ollama_status['status'],
                'host': ollama_status.get('host', Config.OLLAMA_HOST),
                'embedding_model': Config.EMBEDDING_MODEL,
                'llm_model': Config.LLM_MODEL,
                'available_models': ollama_status.get('available_models', [])
            },
            'vector_db': {
                'status': vector_db_status['status'],
                'vector_count': vector_db_status.get('vector_count', 0),
                'dimension': vector_db_status.get('dimension', Config.VECTOR_DIMENSION),
                'index_type': vector_db_status.get('index_type', 'unknown'),
                'uri': Config.MILVUS_URI,
                'collection': Config.COLLECTION_NAME
            },
            'knowledge_base': kb_info,
            'retrieval': {
                'hybrid_enabled': USE_HYBRID_RETRIEVAL,
                'alpha': Config.HYBRID_ALPHA,
                'bm25_weight': Config.BM25_WEIGHT,
                'rrf_k': Config.RRF_K,
                'bm25_k1': Config.BM25_K1,
                'bm25_b': Config.BM25_B
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============ 文档上传 API ============

@api_bp.route('/api/upload', methods=['POST'])
def upload_document():
    """
    上传文档并构建知识库

    接收 .md 或 .txt 文件，进行解析、分段、向量化、存储

    请求:
        POST /api/upload
        Content-Type: multipart/form-data
        file: 文件对象
        chunk_size: 分段大小（可选）
        chunk_overlap: 分段重叠（可选）

    响应:
        {
            "status": "success",
            "filename": "example.md",
            "total_chars": 5000,
            "chunk_count": 5,
            "elapsed_time": 10.5
        }
    """
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有上传文件'
            }), 400

        file = request.files['file']

        # 检查文件名
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '文件名为空'
            }), 400

        # 检查文件扩展名
        allowed = Config.ALLOWED_EXTENSIONS
        ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
        if ext not in allowed:
            return jsonify({
                'status': 'error',
                'message': f'不支持的文件格式，只支持: {", ".join(allowed)}'
            }), 400

        # 获取可选参数
        chunk_size = request.form.get('chunk_size', type=int)
        chunk_overlap = request.form.get('chunk_overlap', type=int)

        # 确保上传目录存在
        upload_folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            Config.UPLOAD_FOLDER
        )
        os.makedirs(upload_folder, exist_ok=True)

        # 保存文件
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # 重新打开文件用于处理
        file.seek(0)

        # 构建知识库
        rag = get_rag_service()
        result = rag.build_knowledge_base(
            file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============ 问答 API ============

@api_bp.route('/api/query', methods=['POST'])
def query():
    """
    基于知识库的问答

    请求:
        POST /api/query
        Content-Type: application/json
        {
            "question": "用户问题",
            "top_k": 5,          // 可选，检索的分段数量
            "stream": false,      // 可选，是否流式返回
            "history": [          // 可选，对话历史
                {"role": "user", "content": "问题1"},
                {"role": "assistant", "content": "回答1"}
            ]
        }

    响应:
        {
            "status": "success",
            "question": "用户问题",
            "answer": "LLM 回答",
            "relevant_chunks": [...],
            "elapsed_time": 5.2
        }
    """
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少 question 参数'
            }), 400

        question = data['question']
        top_k = data.get('top_k', Config.TOP_K)
        stream = data.get('stream', False)
        history = data.get('history', [])  # 获取对话历史

        if stream:
            # 流式响应
            return _stream_query(question, top_k, history)
        else:
            # 普通响应
            rag = get_rag_service()
            result = rag.query(question, top_k=top_k, history=history)

            return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def _stream_query(question: str, top_k: int, history: list = None):
    """
    流式问答 - SSE (Server-Sent Events) 方式

    支持实时显示每一步的处理进度：
    1. 意图识别
    2. 问题向量化
    3. 知识库检索
    4. 上下文构建
    5. LLM 生成回答

    Args:
        question: 当前问题
        top_k: 检索数量
        history: 对话历史列表 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    history = history or []  # 默认空列表

    def generate():
        try:
            rag = get_rag_service()

            # 使用 local_question 避免闭包问题
            local_question = question

            # ========== 步骤 0: 意图识别 ==========
            yield json.dumps({
                'type': 'status',
                'step': 0,
                'step_name': '意图识别',
                'message': '[步骤 1/6] 正在分析问题意图...'
            }) + '\n'

            try:
                intent_result = rag.ollama.classify_intent(local_question)
                intent = intent_result.get('intent', 'KNOWLEDGE_QA')
                confidence = intent_result.get('confidence', 0.5)
                reasoning = intent_result.get('reasoning', '')
                suggested_rewrite = intent_result.get('suggested_rewrite', local_question)

                yield json.dumps({
                    'type': 'status',
                    'step': 0,
                    'step_name': '意图识别',
                    'message': f'[步骤 1/6] 意图识别完成: {intent} (置信度: {confidence:.2f})',
                    'detail': f'分析依据: {reasoning}',
                    'done': True
                }) + '\n'

                # 处理后续问题
                if intent == 'CHITCHAT':
                    yield json.dumps({
                        'type': 'chitchat',
                        'answer': '你好！有什么我可以帮助你的吗？如果你有任何问题需要从知识库中查找答案，请随时告诉我。',
                        'intent': intent
                    }) + '\n'
                    return

                if intent == 'OUT_OF_SCOPE':
                    yield json.dumps({
                        'type': 'chitchat',
                        'answer': '抱歉，这个问题超出了我能够回答的范围。请尝试询问与知识库内容相关的问题。',
                        'intent': intent
                    }) + '\n'
                    return

                if intent == 'FOLLOWUP_QUERY' and suggested_rewrite:
                    local_question = suggested_rewrite
                    yield json.dumps({
                        'type': 'status',
                        'step': 0,
                        'step_name': '意图识别',
                        'message': f'[步骤 1/6] 改写问题: {suggested_rewrite}',
                        'detail': '问题已根据上下文进行改写',
                        'done': True
                    }) + '\n'

            except Exception as e:
                yield json.dumps({
                    'type': 'status',
                    'step': 0,
                    'step_name': '意图识别',
                    'message': f'[!] 意图识别失败，使用默认模式',
                    'detail': str(e),
                    'done': True
                }) + '\n'

            # ========== 步骤 1: 问题向量化 ==========
            yield json.dumps({
                'type': 'status',
                'step': 1,
                'step_name': '问题向量化',
                'message': '[步骤 2/6] 正在将问题转换为向量...'
            }) + '\n'

            query_vector = rag.ollama.embed_text(local_question)

            yield json.dumps({
                'type': 'status',
                'step': 1,
                'step_name': '问题向量化',
                'message': '[步骤 2/6] 向量转换完成',
                'detail': f'向量维度: {len(query_vector)}',
                'done': True
            }) + '\n'

            # ========== 步骤 2: 知识库检索 ==========
            retrieval_mode = 'Hybrid (BM25 + Vector)' if USE_HYBRID_RETRIEVAL else 'Vector Search'
            yield json.dumps({
                'type': 'status',
                'step': 2,
                'step_name': '知识库检索',
                'message': f'[步骤 3/6] 正在检索知识库... ({retrieval_mode})'
            }) + '\n'

            # 检索相关分段
            if USE_HYBRID_RETRIEVAL and rag.hybrid_retriever:
                # Hybrid Retrieval
                yield json.dumps({
                    'type': 'status',
                    'step': 2,
                    'step_name': '知识库检索',
                    'message': '[步骤 3/6] 执行 BM25 关键词检索...'
                }) + '\n'

                bm25_results = rag.hybrid_retriever.search_bm25_only(local_question, top_k=top_k * 2)

                yield json.dumps({
                    'type': 'status',
                    'step': 2,
                    'step_name': '知识库检索',
                    'message': f'[步骤 3/6] BM25 检索完成，找到 {len(bm25_results)} 条结果'
                }) + '\n'

                yield json.dumps({
                    'type': 'status',
                    'step': 2,
                    'step_name': '知识库检索',
                    'message': '[步骤 3/6] 执行向量相似度检索...'
                }) + '\n'

                search_results = rag.hybrid_retriever.search(
                    query=local_question,
                    query_vector=query_vector,
                    top_k=top_k * 2,
                    use_rrf=True,
                    use_rerank=False  # 先不重排序，在单独步骤中进行
                )

                yield json.dumps({
                    'type': 'status',
                    'step': 2,
                    'step_name': '知识库检索',
                    'message': '[步骤 3/6] 执行 RRF 融合排序...'
                }) + '\n'

                # 过滤 - 使用 score 字段（RRF/加权融合返回的是 score，不是 hybrid_score）
                relevant_chunks = [
                    r for r in search_results
                    if r.get('score', 0) > 0 or r.get('hybrid_score', 0) > 0
                ]
            else:
                # Vector Only
                search_results = rag.vector_db.search_vectors(query_vector, limit=top_k * 2)
                relevant_chunks = [
                    r for r in search_results
                    if r.get('score', 0) >= Config.SIMILARITY_THRESHOLD
                ]

            yield json.dumps({
                'type': 'status',
                'step': 2,
                'step_name': '知识库检索',
                'message': f'[步骤 3/6] 初步检索完成，找到 {len(relevant_chunks)} 条候选分段',
                'detail': f'模式: {retrieval_mode}',
                'done': True
            }) + '\n'

            # ========== 步骤 3: CrossEncoder 重排序 ==========
            if rag.hybrid_retriever and rag.hybrid_retriever.reranker:
                yield json.dumps({
                    'type': 'status',
                    'step': 3,
                    'step_name': '本地Rerank',
                    'message': '[步骤 4/6] 正在使用本地 Reranker 进行精排...'
                }) + '\n'

                # 使用 CrossEncoder 重排序
                reranked = rag.hybrid_retriever.reranker.rerank(
                    query=local_question,
                    candidates=relevant_chunks[:top_k * 2],
                    top_k=Config.RERANK_TOP_K,
                    threshold=Config.RERANK_THRESHOLD,
                    query_vector=query_vector
                )

                # 添加重排序标记
                for r in reranked:
                    r['rerank_applied'] = True
                    r['retrieval_type'] = 'hybrid_rrf_rerank'

                # 截取最终结果
                relevant_chunks = reranked[:top_k]

                yield json.dumps({
                    'type': 'status',
                    'step': 3,
                    'step_name': '本地Rerank',
                    'message': f'[步骤 4/6] 本地 Reranker 重排序完成，保留 {len(relevant_chunks)} 条最相关结果',
                    'detail': f'权重: 向量={rag.hybrid_retriever.reranker.vector_weight}, 关键词={rag.hybrid_retriever.reranker.keyword_weight}',
                    'done': True
                }) + '\n'
            else:
                relevant_chunks = relevant_chunks[:top_k]
                yield json.dumps({
                    'type': 'status',
                    'step': 3,
                    'step_name': '本地Rerank',
                    'message': '[步骤 4/6] Reranker 未启用，跳过重排序',
                    'detail': '如需启用，请在配置中设置 ENABLE_CROSS_ENCODER_RERANK = True',
                    'done': True
                }) + '\n'

            # 发送检索到的分段
            yield json.dumps({
                'type': 'chunks',
                'data': relevant_chunks
            }) + '\n'

            # ========== 步骤 4: 上下文构建 ==========
            yield json.dumps({
                'type': 'status',
                'step': 4,
                'step_name': '上下文构建',
                'message': '[步骤 5/6] 正在构建 Prompt 上下文...'
            }) + '\n'

            # 构建上下文
            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                score = chunk.get('hybrid_score', chunk.get('score', 0))
                context_parts.append(
                    f"【文档 {i + 1}】(来源: {chunk['doc_name']}, 分数: {score:.2f})\n{chunk['text']}"
                )
            context = "\n\n".join(context_parts)

            user_prompt = Config.USER_PROMPT_TEMPLATE.format(
                context=context if context else "（知识库中未找到相关内容）",
                question=local_question
            )

            yield json.dumps({
                'type': 'status',
                'step': 4,
                'step_name': '上下文构建',
                'message': '[步骤 5/6] Prompt 构建完成',
                'detail': f'上下文长度: {len(context)} 字符',
                'done': True
            }) + '\n'

            # ========== 步骤 5: LLM 生成 ==========
            yield json.dumps({
                'type': 'status',
                'step': 5,
                'step_name': 'LLM 生成',
                'message': '[步骤 6/6] 正在生成回答...'
            }) + '\n'

            # 构建消息列表，包含历史记录
            messages = []

            # 添加历史对话
            for msg in history:
                if msg.get('role') == 'user':
                    # 历史用户消息：使用原始问题
                    messages.append({'role': 'user', 'content': msg.get('content', '')})
                elif msg.get('role') == 'assistant':
                    # 历史助手消息
                    messages.append({'role': 'assistant', 'content': msg.get('content', '')})

            # 添加当前用户消息（包含上下文的 Prompt）
            messages.append({'role': 'user', 'content': user_prompt})

            print(f"[_stream_query] 对话消息数量: {len(messages)}, 历史轮次: {len(history) // 2}")

            full_answer = []
            chunk_count = 0
            for chunk in rag.ollama.stream_chat(
                messages,
                system_prompt=Config.SYSTEM_PROMPT,
                temperature=0.7
            ):
                content = chunk.get('message', {}).get('content', '')
                chunk_count += 1
                
                # 只发送非空内容
                if content:
                    full_answer.append(content)
                    yield json.dumps({
                        'type': 'content',
                        'content': content
                    }) + '\n'
            
            print(f"[_stream_query] 共处理 {chunk_count} 个 chunk，完整回答长度: {len(''.join(full_answer))}")

            # 完成
            yield json.dumps({
                'type': 'done',
                'answer': ''.join(full_answer),
                'relevant_chunks': relevant_chunks,
                'total_time': '生成完成'
            }) + '\n'

        except Exception as e:
            yield json.dumps({
                'type': 'error',
                'message': str(e)
            }) + '\n'

    return Response(
        generate(),
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


# ============ 知识库管理 API ============

@api_bp.route('/api/knowledge-base')
def get_knowledge_base():
    """
    获取知识库内容

    返回知识库中的所有分段数据
    """
    try:
        rag = get_rag_service()
        result = rag.get_knowledge_base()

        # 转换数据格式以匹配前端期望
        # 前端期望: { chunks: [...], documents: [...] }
        chunks = result.get('data', [])
        documents = result.get('documents', [])

        print(f"[API] /api/knowledge-base 返回: chunks={len(chunks)}, documents={documents}")
        if chunks:
            print(f"[API] 第一个 chunk 结构: {list(chunks[0].keys())}")

        return jsonify({
            'status': 'success',
            'chunks': chunks,
            'documents': documents,
            'total_entities': result.get('total_entities', len(chunks)),
            'document_count': result.get('document_count', len(documents))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/api/knowledge-base/clear', methods=['POST'])
def clear_knowledge_base():
    """
    清空知识库

    可选参数:
        doc_name: 如果指定，只删除该文档
    """
    try:
        data = request.get_json() or {}
        doc_name = data.get('doc_name')

        rag = get_rag_service()
        result = rag.clear_knowledge_base(doc_name=doc_name)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/api/knowledge-base/<path:doc_name>', methods=['DELETE'])
def delete_document(doc_name):
    """
    删除指定文档的所有分段
    """
    try:
        rag = get_rag_service()
        result = rag.clear_knowledge_base(doc_name=doc_name)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============ 配置 API ============

@api_bp.route('/api/config')
def get_config():
    """
    获取当前配置
    """
    return jsonify({
        'status': 'success',
        'config': {
            'chunk_size': Config.DEFAULT_CHUNK_SIZE,
            'chunk_overlap': Config.DEFAULT_CHUNK_OVERLAP,
            'separators': Config.DEFAULT_SEPARATORS,
            'top_k': Config.TOP_K,
            'similarity_threshold': Config.SIMILARITY_THRESHOLD,
            'embedding_model': Config.EMBEDDING_MODEL,
            'llm_model': Config.LLM_MODEL,
            'allowed_extensions': list(Config.ALLOWED_EXTENSIONS),
            'hybrid_retrieval': {
                'enabled': Config.ENABLE_HYBRID_RETRIEVAL,
                'alpha': Config.HYBRID_ALPHA,
                'bm25_weight': Config.BM25_WEIGHT,
                'rrf_k': Config.RRF_K,
                'bm25_k1': Config.BM25_K1,
                'bm25_b': Config.BM25_B
            }
        }
    })


# ============ 错误处理 ============

@api_bp.errorhandler(404)
def not_found(error):
    """404 错误处理"""
    return jsonify({
        'status': 'error',
        'message': '接口不存在'
    }), 404


@api_bp.errorhandler(500)
def internal_error(error):
    """500 错误处理"""
    return jsonify({
        'status': 'error',
        'message': '服务器内部错误'
    }), 500


# ============ 对话管理 API ============

@api_bp.route('/api/conversations', methods=['GET'])
def list_conversations():
    """
    获取对话列表

    查询参数:
        limit: 返回数量限制 (默认 50)
        offset: 偏移量 (默认 0)

    响应:
        {
            "status": "success",
            "conversations": [...],
            "total": 100
        }
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        conv_service = get_conversation_service()
        conversations = conv_service.list_conversations(limit=limit, offset=offset)
        total = conv_service.get_conversation_count()

        return jsonify({
            'status': 'success',
            'conversations': conversations,
            'total': total
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/api/conversations', methods=['POST'])
def create_conversation():
    """
    创建新对话

    请求体:
        {
            "title": "对话标题" (可选),
            "first_message": "第一条消息" (可选，用于生成标题)
        }

    响应:
        {
            "status": "success",
            "conversation_id": 1
        }
    """
    try:
        data = request.get_json() or {}
        title = data.get('title')
        first_message = data.get('first_message')

        conv_service = get_conversation_service()
        conversation_id = conv_service.create_conversation(title=title, first_message=first_message)

        return jsonify({
            'status': 'success',
            'conversation_id': conversation_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/api/conversations/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """
    获取对话详情

    响应:
        {
            "status": "success",
            "conversation": {
                "id": 1,
                "title": "...",
                "messages": [...],
                ...
            }
        }
    """
    try:
        conv_service = get_conversation_service()
        conversation = conv_service.get_conversation(conversation_id)

        if not conversation:
            return jsonify({
                'status': 'error',
                'message': '对话不存在'
            }), 404

        return jsonify({
            'status': 'success',
            'conversation': conversation
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/api/conversations/<int:conversation_id>', methods=['PUT'])
def update_conversation(conversation_id):
    """
    更新对话

    请求体:
        {
            "title": "新标题"
        }

    响应:
        {
            "status": "success"
        }
    """
    try:
        data = request.get_json() or {}

        conv_service = get_conversation_service()

        if 'title' in data:
            success = conv_service.update_conversation_title(conversation_id, data['title'])
            if not success:
                return jsonify({
                    'status': 'error',
                    'message': '对话不存在'
                }), 404

        return jsonify({
            'status': 'success'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/api/conversations/<int:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """
    删除对话

    响应:
        {
            "status": "success"
        }
    """
    try:
        conv_service = get_conversation_service()
        success = conv_service.delete_conversation(conversation_id)

        if not success:
            return jsonify({
                'status': 'error',
                'message': '对话不存在'
            }), 404

        return jsonify({
            'status': 'success'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/api/conversations/<int:conversation_id>/messages', methods=['POST'])
def add_message_to_conversation(conversation_id):
    """
    向对话添加消息

    请求体:
        {
            "role": "user" | "assistant",
            "content": "消息内容",
            "metadata": {...} (可选)
        }

    响应:
        {
            "status": "success",
            "message_id": 123
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'message': '缺少消息内容'
            }), 400

        role = data.get('role')
        content = data.get('content')
        metadata = data.get('metadata', {})

        if not role or not content:
            return jsonify({
                'status': 'error',
                'message': '缺少 role 或 content'
            }), 400

        conv_service = get_conversation_service()
        message_id = conv_service.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata
        )

        return jsonify({
            'status': 'success',
            'message_id': message_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/api/conversations/search', methods=['GET'])
def search_conversations():
    """
    搜索对话

    查询参数:
        q: 搜索关键词
        limit: 返回数量限制 (默认 20)

    响应:
        {
            "status": "success",
            "conversations": [...]
        }
    """
    try:
        keyword = request.args.get('q', '')
        limit = request.args.get('limit', 20, type=int)

        if not keyword:
            return jsonify({
                'status': 'success',
                'conversations': []
            })

        conv_service = get_conversation_service()
        conversations = conv_service.search_conversations(keyword, limit=limit)

        return jsonify({
            'status': 'success',
            'conversations': conversations
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============ 增强版问答 API（支持对话 ID） ============

@api_bp.route('/api/query-with-conversation', methods=['POST'])
def query_with_conversation():
    """
    基于知识库的问答（关联对话版本）

    支持将问答自动追加到指定对话中。

    请求:
        POST /api/query-with-conversation
        Content-Type: application/json
        {
            "question": "用户问题",
            "conversation_id": 1,       // 对话 ID (可选，不传则创建新对话)
            "top_k": 5,                  // 可选，检索的分段数量
            "stream": true,              // 可选，是否流式返回
            "history": [...]             // 可选，历史消息（用于 LLM 上下文）
        }

    响应:
        {
            "status": "success",
            "conversation_id": 1,
            "question_id": 123,
            "answer_id": 124,
            "answer": "LLM 回答",
            "relevant_chunks": [...],
            "elapsed_time": 5.2
        }
    """
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少 question 参数'
            }), 400

        question = data['question']
        conversation_id = data.get('conversation_id')
        top_k = data.get('top_k', Config.TOP_K)
        stream = data.get('stream', False)
        history = data.get('history', [])

        conv_service = get_conversation_service()

        # 如果没有指定对话 ID，创建新对话
        if not conversation_id:
            conversation_id = conv_service.create_conversation(first_message=question)

        # 添加用户消息到数据库
        conv_service.add_message(
            conversation_id=conversation_id,
            role='user',
            content=question
        )

        # 如果是流式响应
        if stream:
            return _stream_query_with_conversation(question, top_k, history, conversation_id)
        else:
            # 普通响应
            rag = get_rag_service()
            result = rag.query(question, top_k=top_k, history=history)

            # 添加助手回复到数据库
            conv_service.add_message(
                conversation_id=conversation_id,
                role='assistant',
                content=result.get('answer', '')
            )

            return jsonify({
                'status': 'success',
                'conversation_id': conversation_id,
                **result
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def _stream_query_with_conversation(question: str, top_k: int, history: list, conversation_id: int):
    """
    流式问答（关联对话版本）

    Args:
        question: 当前问题
        top_k: 检索数量
        history: 对话历史（用于 LLM 上下文）
        conversation_id: 对话 ID
    """
    history = history or []

    def generate():
        try:
            rag = get_rag_service()
            conv_service = get_conversation_service()
            local_question = question

            # ========== 步骤 0: 意图识别 ==========
            yield json.dumps({
                'type': 'status',
                'step': 0,
                'step_name': '意图识别',
                'message': '[步骤 1/6] 正在分析问题意图...'
            }) + '\n'

            try:
                intent_result = rag.ollama.classify_intent(local_question)
                intent = intent_result.get('intent', 'KNOWLEDGE_QA')
                confidence = intent_result.get('confidence', 0.5)
                reasoning = intent_result.get('reasoning', '')
                suggested_rewrite = intent_result.get('suggested_rewrite', local_question)

                yield json.dumps({
                    'type': 'status',
                    'step': 0,
                    'step_name': '意图识别',
                    'message': f'[步骤 1/6] 意图识别完成: {intent} (置信度: {confidence:.2f})',
                    'detail': f'分析依据: {reasoning}',
                    'done': True
                }) + '\n'

                if intent == 'CHITCHAT':
                    conv_service.add_message(
                        conversation_id=conversation_id,
                        role='assistant',
                        content='你好！有什么我可以帮助你的吗？如果你有任何问题需要从知识库中查找答案，请随时告诉我。'
                    )
                    yield json.dumps({
                        'type': 'chitchat',
                        'answer': '你好！有什么我可以帮助你的吗？如果你有任何问题需要从知识库中查找答案，请随时告诉我。',
                        'intent': intent
                    }) + '\n'
                    return

                if intent == 'OUT_OF_SCOPE':
                    conv_service.add_message(
                        conversation_id=conversation_id,
                        role='assistant',
                        content='抱歉，这个问题超出了我能够回答的范围。请尝试询问与知识库内容相关的问题。'
                    )
                    yield json.dumps({
                        'type': 'chitchat',
                        'answer': '抱歉，这个问题超出了我能够回答的范围。请尝试询问与知识库内容相关的问题。',
                        'intent': intent
                    }) + '\n'
                    return

                if intent == 'FOLLOWUP_QUERY' and suggested_rewrite:
                    local_question = suggested_rewrite
                    yield json.dumps({
                        'type': 'status',
                        'step': 0,
                        'step_name': '意图识别',
                        'message': f'[步骤 1/6] 改写问题: {suggested_rewrite}',
                        'detail': '问题已根据上下文进行改写',
                        'done': True
                    }) + '\n'

            except Exception as e:
                yield json.dumps({
                    'type': 'status',
                    'step': 0,
                    'step_name': '意图识别',
                    'message': f'[!] 意图识别失败，使用默认模式',
                    'detail': str(e),
                    'done': True
                }) + '\n'

            # ========== 步骤 1-5: 与原有流式处理相同 ==========
            # 问题向量化
            yield json.dumps({
                'type': 'status',
                'step': 1,
                'step_name': '问题向量化',
                'message': '[步骤 2/6] 正在将问题转换为向量...'
            }) + '\n'

            query_vector = rag.ollama.embed_text(local_question)

            yield json.dumps({
                'type': 'status',
                'step': 1,
                'step_name': '问题向量化',
                'message': '[步骤 2/6] 向量转换完成',
                'detail': f'向量维度: {len(query_vector)}',
                'done': True
            }) + '\n'

            # 知识库检索
            retrieval_mode = 'Hybrid (BM25 + Vector)' if USE_HYBRID_RETRIEVAL else 'Vector Search'
            yield json.dumps({
                'type': 'status',
                'step': 2,
                'step_name': '知识库检索',
                'message': f'[步骤 3/6] 正在检索知识库... ({retrieval_mode})'
            }) + '\n'

            if USE_HYBRID_RETRIEVAL and rag.hybrid_retriever:
                bm25_results = rag.hybrid_retriever.search_bm25_only(local_question, top_k=top_k * 2)

                search_results = rag.hybrid_retriever.search(
                    query=local_question,
                    query_vector=query_vector,
                    top_k=top_k * 2,
                    use_rrf=True,
                    use_rerank=False
                )

                relevant_chunks = [
                    r for r in search_results
                    if r.get('score', 0) > 0 or r.get('hybrid_score', 0) > 0
                ]
            else:
                search_results = rag.vector_db.search_vectors(query_vector, limit=top_k * 2)
                relevant_chunks = [
                    r for r in search_results
                    if r.get('score', 0) >= Config.SIMILARITY_THRESHOLD
                ]

            yield json.dumps({
                'type': 'status',
                'step': 2,
                'step_name': '知识库检索',
                'message': f'[步骤 3/6] 初步检索完成，找到 {len(relevant_chunks)} 条候选分段',
                'detail': f'模式: {retrieval_mode}',
                'done': True
            }) + '\n'

            # CrossEncoder 重排序
            if rag.hybrid_retriever and rag.hybrid_retriever.reranker:
                yield json.dumps({
                    'type': 'status',
                    'step': 3,
                    'step_name': '本地Rerank',
                    'message': '[步骤 4/6] 正在使用本地 Reranker 进行精排...'
                }) + '\n'

                reranked = rag.hybrid_retriever.reranker.rerank(
                    query=local_question,
                    candidates=relevant_chunks[:top_k * 2],
                    top_k=Config.RERANK_TOP_K,
                    threshold=Config.RERANK_THRESHOLD,
                    query_vector=query_vector
                )

                for r in reranked:
                    r['rerank_applied'] = True
                    r['retrieval_type'] = 'hybrid_rrf_rerank'

                relevant_chunks = reranked[:top_k]

                yield json.dumps({
                    'type': 'status',
                    'step': 3,
                    'step_name': '本地Rerank',
                    'message': f'[步骤 4/6] 本地 Reranker 重排序完成，保留 {len(relevant_chunks)} 条最相关结果',
                    'detail': f'权重: 向量={rag.hybrid_retriever.reranker.vector_weight}, 关键词={rag.hybrid_retriever.reranker.keyword_weight}',
                    'done': True
                }) + '\n'
            else:
                relevant_chunks = relevant_chunks[:top_k]
                yield json.dumps({
                    'type': 'status',
                    'step': 3,
                    'step_name': '本地Rerank',
                    'message': '[步骤 4/6] Reranker 未启用，跳过重排序',
                    'detail': '如需启用，请在配置中设置 ENABLE_CROSS_ENCODER_RERANK = True',
                    'done': True
                }) + '\n'

            # 发送检索到的分段
            yield json.dumps({
                'type': 'chunks',
                'data': relevant_chunks
            }) + '\n'

            # 上下文构建
            yield json.dumps({
                'type': 'status',
                'step': 4,
                'step_name': '上下文构建',
                'message': '[步骤 5/6] 正在构建 Prompt 上下文...'
            }) + '\n'

            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                score = chunk.get('hybrid_score', chunk.get('score', 0))
                context_parts.append(
                    f"【文档 {i + 1}】(来源: {chunk['doc_name']}, 分数: {score:.2f})\n{chunk['text']}"
                )
            context = "\n\n".join(context_parts)

            user_prompt = Config.USER_PROMPT_TEMPLATE.format(
                context=context if context else "（知识库中未找到相关内容）",
                question=local_question
            )

            yield json.dumps({
                'type': 'status',
                'step': 4,
                'step_name': '上下文构建',
                'message': '[步骤 5/6] Prompt 构建完成',
                'detail': f'上下文长度: {len(context)} 字符',
                'done': True
            }) + '\n'

            # LLM 生成
            yield json.dumps({
                'type': 'status',
                'step': 5,
                'step_name': 'LLM 生成',
                'message': '[步骤 6/6] 正在生成回答...'
            }) + '\n'

            # 构建消息列表
            messages = []
            for msg in history:
                if msg.get('role') == 'user':
                    messages.append({'role': 'user', 'content': msg.get('content', '')})
                elif msg.get('role') == 'assistant':
                    messages.append({'role': 'assistant', 'content': msg.get('content', '')})
            messages.append({'role': 'user', 'content': user_prompt})

            full_answer = []
            chunk_count = 0
            for chunk in rag.ollama.stream_chat(
                messages,
                system_prompt=Config.SYSTEM_PROMPT,
                temperature=0.7
            ):
                content = chunk.get('message', {}).get('content', '')
                chunk_count += 1

                if content:
                    full_answer.append(content)
                    yield json.dumps({
                        'type': 'content',
                        'content': content,
                        'conversation_id': conversation_id
                    }) + '\n'

            final_answer = ''.join(full_answer)

            # 将助手回复保存到数据库
            conv_service.add_message(
                conversation_id=conversation_id,
                role='assistant',
                content=final_answer
            )

            # 完成
            yield json.dumps({
                'type': 'done',
                'conversation_id': conversation_id,
                'answer': final_answer,
                'relevant_chunks': relevant_chunks,
                'total_time': '生成完成'
            }) + '\n'

        except Exception as e:
            yield json.dumps({
                'type': 'error',
                'message': str(e)
            }) + '\n'

    return Response(
        generate(),
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )
