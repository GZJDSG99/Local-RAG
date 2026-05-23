# Local RAG - 本地知识库问答系统

一个基于本地大模型和向量检索的 RAG (Retrieval-Augmented Generation) 知识库问答系统。完全本地化部署，保护数据隐私，适合学习和生产使用。

## 主要特性

- **完全本地化** - 无需云服务，数据不出本地
- **Hybrid 检索** - 结合 BM25 关键词检索和向量语义检索
- **意图识别** - 智能区分问答、闲聊、追问、系统命令
- **本地 Rerank** - 使用 Ollama 本地 Reranker 进行精排
- **流式输出** - 实时展示 6 步处理流程和生成过程
- **多轮对话** - 支持对话历史上下文连贯
- **对话管理** - 保存和管理历史对话
- **现代 UI** - 深浅主题、响应式布局、可折叠侧边栏

## 技术栈

| 组件 | 技术 | 说明 |
| --- | --- | --- |
| **LLM 推理** | Ollama | 本地运行开源大语言模型 |
| **向量模型** | Qwen3-Embedding-0.6B-f16 | 文本向量化（1024 维） |
| **Reranker** | Ollama 本地模型 | 本地 CrossEncoder 重排序 |
| **向量数据库** | Milvus | 存储和检索向量数据 |
| **Web 框架** | Flask | Python Web 服务 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户界面 (Web)                              │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │
│   │  智能问答  │   │  知识库   │   │  系统设置  │   │     侧边栏       │  │
│   │          │   │          │   │          │   │  • 历史对话       │  │
│   │ • 流程指示 │   │ • 文档上传 │   │ • 模型配置 │   │  • 系统状态      │  │
│   │ • 流式显示 │   │ • 检索查看 │   │ • 检索配置 │   │  • 新建对话      │  │
│   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └────────┬────────┘  │
└─────────┼───────────────┼───────────────┼─────────────────┼──────────┘
          │               │               │                 │
          └───────────────┴───────────────┴─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │     Flask API       │
                    │  /api/query-with-   │
                    │   conversation      │
                    └──────────┬──────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                              │                                      │
▼                              ▼                                      ▼
┌──────────────┐    ┌──────────────────────┐    ┌────────────────────┐
│   意图识别    │    │      RAG 服务        │    │    对话管理        │
│              │    │                      │    │                    │
│ • 知识库问答  │    │ 1. BM25 关键词检索   │    │ • 创建对话         │
│ • 闲聊对话   │    │ 2. 向量语义检索      │    │ • 保存消息         │
│ • 追问改写   │    │ 3. RRF 融合排序      │    │ • 加载历史         │
│ • 系统命令   │    │ 4. 本地 Rerank 精排  │    │ • 对话搜索         │
│ • 超范围检测 │    │ 5. 上下文构建        │    │                    │
│              │    │ 6. LLM 流式生成      │    │                    │
└──────┬───────┘    └──────────┬───────────┘    └────────────────────┘
       │                        │
       │            ┌────────────┴────────────┐
       │            │                         │
       ▼            ▼                         ▼
┌────────────┐  ┌────────────┐  ┌────────────────────┐
│   Ollama   │  │   Milvus   │  │   本地存储          │
│            │  │            │  │                    │
│ • Embedding│  │ • 向量存储 │  │ • conversations.db │
│ • LLM 生成 │  │ • HNSW 索引│  │ • uploads/         │
│ • Reranker │  │            │  │ • config.json      │
└────────────┘  └────────────┘  └────────────────────┘
```

## 目录结构

```
local-RAG/
├── app/
│   ├── __init__.py              # Flask 应用工厂
│   ├── config.py                # 配置管理
│   ├── document_parser.py       # 文档解析（Markdown/TXT）
│   ├── text_chunker.py          # 文本分段
│   ├── ollama_client.py         # Ollama API 封装
│   ├── milvus_client.py         # Milvus 向量数据库封装
│   ├── faiss_client.py          # FAISS 向量索引（可选）
│   ├── hybrid_retriever.py      # Hybrid 检索（BM25 + Vector）
│   ├── reranker.py              # 本地 Reranker
│   ├── intent_detector.py       # 意图识别
│   ├── rag_service.py           # RAG 核心服务
│   ├── conversation_service.py  # 对话管理服务
│   └── routes.py                # Flask 路由定义
├── static/
│   ├── css/
│   │   └── style.css           # 样式（深浅主题、响应式）
│   └── js/
│       └── app.js               # 前端交互逻辑
├── templates/
│   └── index.html               # 单页应用
├── uploads/                      # 上传文件存储
├── conversations.db              # 对话历史数据库
├── config.json                   # 用户配置文件
├── run.py                        # 应用入口
├── requirements.txt              # Python 依赖
└── README.md                     # 项目文档
```

## 快速开始

### 前置要求

- Python 3.8+
- Ollama（已安装并运行）
- Milvus（Docker 部署）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 拉取模型

```bash
# 向量模型
ollama pull Qwen3-Embedding-0.6B-f16:latest

# LLM 模型
ollama pull gemma3:1b

# Reranker 模型（可选，用于重排序）
ollama pull nomic-embed-text:latest
```

### 3. 启动 Milvus

```bash
docker run -d \
  --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.5.18
```

### 4. 启动应用

```bash
python run.py
```

访问 http://localhost:5000

## API 接口

### 问答接口

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/query-with-conversation` | POST | 关联对话的流式问答（推荐） |
| `/api/query` | POST | 基础问答接口 |

### 对话管理

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/conversations` | GET | 获取对话列表 |
| `/api/conversations` | POST | 创建新对话 |
| `/api/conversations/<id>` | GET | 获取对话详情 |
| `/api/conversations/<id>` | PUT | 更新对话 |
| `/api/conversations/<id>` | DELETE | 删除对话 |
| `/api/conversations/<id>/messages` | POST | 添加消息 |

### 知识库管理

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/upload` | POST | 上传文档 |
| `/api/knowledge-base` | GET | 获取知识库内容 |
| `/api/knowledge-base/<name>` | DELETE | 删除文档 |
| `/api/knowledge-base/clear` | POST | 清空知识库 |

### 系统接口

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/status` | GET | 获取系统状态 |
| `/api/config` | GET | 获取配置信息 |

### 问答接口示例

```bash
curl -X POST http://localhost:5000/api/query-with-conversation \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是机器学习？",
    "top_k": 5,
    "stream": true,
    "history": []
  }'
```

响应（流式 NDJSON）：

```json
{"type": "status", "step": 0, "step_name": "意图识别", "message": "[步骤 1/6] 正在分析问题意图..."}
{"type": "status", "step": 1, "step_name": "问题向量化", "message": "[步骤 2/6] 正在将问题转换为向量..."}
{"type": "status", "step": 2, "step_name": "知识库检索", "message": "[步骤 3/6] 正在检索知识库..."}
{"type": "status", "step": 3, "step_name": "本地Rerank", "message": "[步骤 4/6] 正在使用本地 Reranker 进行精排..."}
{"type": "chunks", "data": [...]}
{"type": "status", "step": 4, "step_name": "上下文构建", "message": "[步骤 5/6] 正在构建 Prompt 上下文..."}
{"type": "status", "step": 5, "step_name": "LLM 生成", "message": "[步骤 6/6] 正在生成回答..."}
{"type": "content", "content": "机器"}
{"type": "content", "content": "学习"}
{"type": "content", "content": "是人工"}
{"type": "done", "answer": "机器学习是人工智能的一个分支..."}
```

## 功能详解

### 智能问答流程

系统采用 6 步处理流程，实时展示每一步的状态：

| 步骤 | 名称 | 说明 |
| --- | --- | --- |
| 1 | 意图识别 | 判断问题类型（知识问答/闲聊/追问/系统命令） |
| 2 | 问题向量化 | 使用 Qwen3-Embedding 将问题转换为向量 |
| 3 | 知识库检索 | Hybrid 检索（BM25 + 向量）获取候选分段 |
| 4 | 本地 Rerank | 使用 Ollama Reranker 进行精排 |
| 5 | 上下文构建 | 组装 Prompt，包含检索结果和对话历史 |
| 6 | LLM 生成 | 流式生成回答 |

### Hybrid 检索

结合两种检索方式的优势：

1. **BM25 检索** - 关键词精确匹配，适合专有名词
2. **向量检索** - 语义相似度匹配，理解上下文
3. **RRF 融合** - 使用 Reciprocal Rank Fusion 合并结果
4. **本地 Rerank** - 使用 Ollama Reranker 进行精排

```
用户问题
    │
    ▼
┌─────────────┐     ┌─────────────┐
│    BM25     │     │   Vector    │
│    检索     │     │    检索     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
    Top-K * 2          Top-K * 2
       │                   │
       └─────────┬─────────┘
                 ▼
          ┌───────────┐
          │    RRF    │
          │    融合   │
          └─────┬─────┘
                │
                ▼
        ┌───────────────┐
        │  本地 Rerank  │
        │    (精排)     │
        └───────┬───────┘
                │
                ▼
           最终 Top-K
```

### 意图识别

系统支持多种意图类型，自动判断处理方式：

| 意图类型 | 特征 | 处理方式 |
| --- | --- | --- |
| KNOWLEDGE_QA | 知识问答 | Hybrid 检索 + LLM 生成 |
| CHITCHAT | 闲聊 | 直接 LLM 回复 |
| FOLLOWUP_QUERY | 追问 | 自动改写 + 检索 |
| SYSTEM_CMD | 系统命令 | 执行对应操作 |
| OUT_OF_SCOPE | 超范围 | 礼貌拒绝 |

### 文本分段策略

```
分段大小: 1024 字符
分段重叠: 100 字符
分隔符优先级: \n#  >  \n##  >  \n###  >  \n\n  >  。  >  \n  >  空格
```

- Markdown 标题优先切分，保持文档结构完整性
- 首尾重叠确保上下文连贯
- 记录分段索引和来源文档便于溯源

## 配置说明

### 配置文件 (config.json)

首次运行会自动生成：

```json
{
  "chunk_size": 1024,
  "chunk_overlap": 100,
  "top_k": 5,
  "model": "gemma3:1b",
  "embedding_model": "Qwen3-Embedding-0.6B-f16:latest",
  "temperature": 0.7,
  "stream": true
}
```

### 配置参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| chunk_size | 1024 | 分段字符数 |
| chunk_overlap | 100 | 分段重叠数 |
| top_k | 5 | 检索返回数量 |
| model | gemma3:1b | LLM 模型 |
| embedding_model | Qwen3-Embedding-0.6B-f16:latest | 向量模型 |
| temperature | 0.7 | 生成温度 |

### Hybrid 检索配置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| HYBRID_ALPHA | 0.5 | BM25/向量权重平衡 |
| BM25_WEIGHT | 0.5 | BM25 最终权重 |
| RRF_K | 60 | RRF 融合参数 |

## 前端界面

### 页面结构

```
┌─────────────────────────────────────────────────────────────┐
│ Local RAG                                              [≡] │
├────────┬────────────────────────────────────────────────────┤
│        │                                                    │
│ 💬 问答 │  ┌──────────────────────────────────────────────┐ │
│ 📚 知识 │  │                                              │ │
│ ⚙ 设置 │  │              对话区域                          │ │
│        │  │                                              │ │
│ ────── │  │  ┌──────────────────────────────────────┐   │ │
│ 历史   │  │  │  用户问题                              │   │ │
│ 对话   │  │  └──────────────────────────────────────┘   │ │
│        │  │  ┌──────────────────────────────────────┐   │ │
│ + 新建 │  │  │  助手回答                              │   │ │
│        │  │  └──────────────────────────────────────┘   │ │
│        │  │                                              │ │
│        │  │  ┌──────────────────────────────────────┐   │ │
│        │  │  │  6 步流程指示器                        │   │ │
│        │  │  │  ○ 意图识别 ○ 向量化 ○ 检索 ...       │   │ │
│        │  │  └──────────────────────────────────────┘   │ │
│        │  └──────────────────────────────────────────────┘ │
│        │  ┌──────────────────────────────────────────────┐ │
│        │  │ [📎] 请输入问题...              [发送问题]    │ │
│        │  └──────────────────────────────────────────────┘ │
└────────┴────────────────────────────────────────────────────┘
```

### 侧边栏模式

| 模式 | 宽度 | 显示内容 |
| --- | --- | --- |
| 展开 | 260px | Logo、历史对话、导航菜单、系统状态 |
| 收起 | 70px | 竖向胶囊菜单（3个图标按钮） |

## 开发指南

### 添加新的意图类型

在 `intent_detector.py` 中扩展意图识别规则：

```python
INTENT_PATTERNS = {
    # 现有模式...
    'new_intent': [
        r'关键词1',
        r'关键词2',
    ]
}
```

### 自定义 LLM

修改 `ollama_client.py` 中的 `generate` 方法：

```python
def generate(self, prompt: str, model: str = None) -> Iterator[str]:
    model = model or self.default_model
    # 调用自定义 LLM
    # ...
```

## 常见问题

### Ollama 连接失败

```bash
# 启动 Ollama 服务
ollama serve

# 检查模型
ollama list
```

### Milvus 连接失败

```bash
# 检查容器状态
docker ps | grep milvus

# 重启容器
docker restart milvus
```

### 检索结果不理想

1. 调整 `top_k` 参数增加检索范围
2. 优化分段策略（减小 chunk_size）
3. 调整 `HYBRID_ALPHA` 参数平衡 BM25/向量权重
4. 检查知识库内容质量

## 学习资源

- [Ollama 官方文档](https://docs.ollama.com/)
- [Milvus 官方文档](https://milvus.io/docs)
- [RAG 技术综述](https://arxiv.org/abs/2312.10997)
- [Qwen3-Embedding 模型](https://ollama.com/library/qwen3-embedding)
- [Gemma 模型文档](https://ai.google.dev/gemma)

## License

MIT License
