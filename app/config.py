"""
应用配置模块
============

管理所有可配置参数，包括 Ollama 连接、Milvus 设置、分段规则等。
"""

import os

class Config:
    """应用配置类"""

    # ============ Flask 配置 ============
    SECRET_KEY = os.environ.get('SECRET_KEY', 'local-rag-secret-key-2026')
    DEBUG = True

    # ============ Ollama 配置 ============
    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

    # 向量模型 - 用于将文本转换为向量
    # Qwen3-Embedding 系列是阿里巴巴开源的高质量嵌入模型
    # 支持 100+ 语言，0.6B/4B/8B 三种规模
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'Qwen3-Embedding-0.6B-f16:latest')

    # LLM 问答模型 - 用于基于上下文生成回答
    # Gemma3 是 Google 开源的轻量级模型，1b 表示 10 亿参数
    LLM_MODEL = os.environ.get('LLM_MODEL', 'gemma3:1b')

    # ============ Milvus 配置 ============
    # Milvus 是一个开源的向量数据库，专门用于存储和检索高维向量
    # 支持数十亿级向量规模，提供多种相似度度量方式

    # Milvus 连接方式：
    # 1. Docker 部署: 'http://localhost:19530'
    # 2. Milvus Lite (嵌入式，无需 Docker): './milvus_lite.db'
    #
    # 推荐：使用 Docker 部署的 Milvus，支持更大规模数据

    # 默认使用 Docker 部署的 Milvus
    MILVUS_URI = os.environ.get('MILVUS_URI', 'http://localhost:19530')

    # 集合名称 - Milvus 中存储向量的容器
    COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'knowledge_base')

    # 向量维度 - Qwen3-Embedding 默认输出 1024 维向量
    # 注意：这个值必须与实际使用的 embedding 模型输出一致
    VECTOR_DIMENSION = 1024

    # ============ 文本分段配置 ============
    # 文本分段是将长文档切分成小块的过程
    # 分段大小的选择会影响检索效果和上下文长度

    # 默认分段大小（字符数）
    # 1024 字符约等于 256-512 个中文词
    DEFAULT_CHUNK_SIZE = 1024

    # 分段重叠大小（字符数）
    # 相邻分段之间保留的重叠内容，有助于保持上下文连贯性
    DEFAULT_CHUNK_OVERLAP = 100

    # 分段分隔符列表（按优先级排序）
    # 系统会优先使用前面的分隔符进行切分
    # Markdown 标题分隔符优先，保持文档结构完整性
    DEFAULT_SEPARATORS = ['\n# ', '\n## ', '\n### ', '\n\n', '。', '\n', ' ']

    # ============ 检索配置 ============
    # 检索时返回的最相关分段数量
    TOP_K = 5

    # 相似度阈值 - 只返回高于此阈值的检索结果
    SIMILARITY_THRESHOLD = 0.5

    # ============ Hybrid Retrieval 配置 ============
    # Hybrid Retrieval = BM25 (关键词精确匹配) + Vector Search (语义相似度)
    # 两种方法互补：BM25 擅长精确词项匹配，向量检索擅长语义理解

    # 是否启用 Hybrid Retrieval
    ENABLE_HYBRID_RETRIEVAL = True

    # BM25 参数
    # k1: term frequency saturation parameter (控制词频对得分的影响)
    # 值越高，词频的影响越大。通常 1.2 ~ 2.0
    BM25_K1 = 1.5

    # b: length normalization parameter (控制文档长度对得分的影响)
    # 值越大，短文档得分相对更高。范围 0 ~ 1，通常 0.75
    BM25_B = 0.75

    # Hybrid Score 权重
    # alpha=0 表示纯 BM25, alpha=1 表示纯向量检索
    # 设为 0.5 表示两种方法同等重要
    HYBRID_ALPHA = 0.5

    # BM25 权重 (0.0 ~ 1.0)，最终权重 = (1 - HYBRID_ALPHA) * BM25_WEIGHT
    BM25_WEIGHT = 0.5

    # RRF (Reciprocal Rank Fusion) 参数
    # 用于合并多个检索结果列表的排名
    RRF_K = 60

    # ============ CrossEncoder Rerank 配置 ============
    # CrossEncoder 是一种更精确的重排序模型
    # 它同时接收 query 和 document，计算相关性分数
    # 可以显著提升检索结果的排序质量

    # 是否启用 CrossEncoder 重排序（首次使用需要下载模型，可能较慢）
    # 现在使用本地 Ollama Reranker，无需下载外部模型
    ENABLE_CROSS_ENCODER_RERANK = True

    # CrossEncoder 模型名称
    # 常用中文重排序模型：
    # - 'BAAI/bge-reranker-base' (推荐，中英文效果都不错)
    # - 'BAAI/bge-reranker-large' (更大更强，但需要更多显存)
    # - 'moka-ai/m3e-base-cross' (中文专用)
    CROSS_ENCODER_MODEL = os.environ.get('CROSS_ENCODER_MODEL', 'BAAI/bge-reranker-base')

    # Rerank 返回数量 (应 >= TOP_K)
    RERANK_TOP_K = 10

    # CrossEncoder 重排序后的最终返回数量
    FINAL_TOP_K = 5

    # 重排序分数阈值 (0-1，越高越严格，0表示不过滤)
    RERANK_THRESHOLD = 0.0

    # ============ 文件上传配置 ============
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 最大 16MB
    ALLOWED_EXTENSIONS = {'txt', 'md'}

    # ============ RAG 提示词配置 ============
    # 这些提示词会指导 LLM 如何基于检索到的上下文回答问题

    SYSTEM_PROMPT = """你是一个知识库问答助手。请根据提供的上下文信息回答用户的问题。

要求：
1. 只使用提供的上下文信息来回答问题
2. 如果上下文中没有相关信息，请如实说明"我没有在知识库中找到相关信息"
3. 回答要准确、简洁、有条理
4. 如果有多条相关信息，可以综合回答"""

    USER_PROMPT_TEMPLATE = """上下文信息：
{context}

用户问题：{question}

请根据以上上下文信息回答用户的问题："""

    # ============ 意图识别配置 ============
    # 意图识别用于判断用户问题是否需要知识库检索

    INTENT_SYSTEM_PROMPT = """你是一个严谨的意图识别助手，专门用于RAG知识问答系统。你的任务是分析用户输入，判断其真实意图，并按指定JSON格式输出结果。"""

    INTENT_USER_PROMPT_TEMPLATE = """## 意图类别（你必须从以下5类中选择且只选一类）
- **KNOWLEDGE_QA**：用户希望从知识库/文档中获取事实、定义、操作步骤、规范、产品参数等客观信息。特征：问题中包含"是什么""怎么""为什么""多少""哪个"等疑问词，或明确要求查找信息。
- **CHITCHAT**：社交性对话，无需知识库支持。特征：问候、感谢、告别、情感表达、无关闲聊（如"你好""谢谢""今天天气真好"）。
- **SYSTEM_CMD**：用户明确要求系统执行非问答类操作。特征：包含"只输出…""不要检索…""改用…模式""清空对话""/"开头的命令。
- **FOLLOWUP_QUERY**：当前问题依赖上文对话才能完整理解。特征：包含指代词（"它""那个""第二个"）、省略主语/宾语、或与上一轮对话强相关。
- **OUT_OF_SCOPE**：超出业务/知识范围或敏感内容。特征：涉及政治敏感、违法信息、与系统支持的主题完全无关。

## 判定优先级（从高到低）
1. 如果是系统命令或明确拒绝回答的内容 → SYSTEM_CMD 或 OUT_OF_SCOPE
2. 如果明显无需任何知识 → CHITCHAT
3. 如果问题中缺少关键实体且必须依赖上文 → FOLLOWUP_QUERY
4. 否则 → KNOWLEDGE_QA

## 输出格式（严格JSON，不要包含任何额外解释或Markdown标记）
{{"intent": "KNOWLEDGE_QA | CHITCHAT | SYSTEM_CMD | FOLLOWUP_QUERY | OUT_OF_SCOPE", "confidence": 0.0-1.0, "reasoning": "简要判断依据（一句话）", "need_retrieval": true/false, "suggested_rewrite": "对KNOWLEDGE_QA/FOLLOWUP_QUERY建议的重写后问题（其他情况为空字符串）"}}

## 示例

用户输入：什么是机器学习？
输出：
{{"intent": "KNOWLEDGE_QA", "confidence": 0.99, "reasoning": "包含明确的定义型疑问词'什么是'，需要知识库支持", "need_retrieval": true, "suggested_rewrite": "机器学习定义"}}

用户输入：你好呀
输出：
{{"intent": "CHITCHAT", "confidence": 1.0, "reasoning": "常见问候语，无需知识库", "need_retrieval": false, "suggested_rewrite": ""}}

用户输入：只输出答案，不要解释。
输出：
{{"intent": "SYSTEM_CMD", "confidence": 1.0, "reasoning": "用户明确给出了输出格式指令", "need_retrieval": false, "suggested_rewrite": ""}}

## 注意
1. 只输出纯JSON对象，不要有```json```标记，不要有任何前后缀说明文字。
2. confidence不得低于0.0或高于1.0。

用户输入：{query}"""

    # 意图识别置信度阈值，低于此值不进行检索
    INTENT_CONFIDENCE_THRESHOLD = 0.6
