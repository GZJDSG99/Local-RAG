"""
Ollama 客户端模块
=================

封装与 Ollama 服务的交互，包括：
1. 文本向量化（使用 embedding 模型）
2. 问答生成（使用 chat 模型）

教学要点:
- API 调用：使用 Python 的 ollama 库调用本地模型服务
- 流式响应：实时显示 LLM 生成的回答
- 模型选择：不同任务使用不同的模型

Ollama 是什么？
- Ollama 是一个本地 LLM 推理框架
- 可以在本地运行各种开源大语言模型
- 提供 REST API 和 Python/JS SDK
- 无需 GPU 云资源，在个人电脑上即可运行

Qwen3-Embedding 模型：
- 阿里巴巴开源的高质量文本嵌入模型
- 将文本转换为固定维度的向量（1024维）
- 支持中英文，语义理解能力强
- 比 OpenAI 的 text-embedding-ada-002 效果更好

Gemma3 模型：
- Google 开源的大语言模型
- 1b 表示 10 亿参数（适合消费级 GPU）
- 支持多种任务：问答、总结、翻译等
"""

import ollama
from typing import List, Dict, Optional, Iterator
import sys
from .config import Config


class OllamaClient:
    """
    Ollama 服务客户端

    封装了对 Ollama API 的调用，提供：
    - embeddings(): 文本向量化
    - chat(): 问答生成
    - stream_chat(): 流式问答生成
    """

    def __init__(
        self,
        host: str = None,
        embedding_model: str = None,
        llm_model: str = None
    ):
        """
        初始化 Ollama 客户端

        Args:
            host: Ollama 服务地址，默认使用 Config 中的配置
            embedding_model: 向量化模型名称
            llm_model: 问答模型名称
        """
        self.host = host or Config.OLLAMA_HOST
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.llm_model = llm_model or Config.LLM_MODEL

        # 设置 ollama 客户端的主机地址
        self._client = ollama.Client(host=self.host)

        print(f"[Ollama 客户端] 初始化完成")
        print(f"  服务地址: {self.host}")
        print(f"  向量模型: {self.embedding_model}")
        print(f"  问答模型: {self.llm_model}")

    def check_connection(self) -> Dict[str, any]:
        """
        检查 Ollama 服务连接状态

        Returns:
            包含连接状态的字典
        """
        try:
            # 尝试获取模型列表来验证连接
            models = self._client.list()
            return {
                'status': 'connected',
                'host': self.host,
                'available_models': [m['name'] for m in models.get('models', [])]
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_embedding_dimension(self) -> int:
        """
        获取向量化模型的输出维度

        Returns:
            向量维度（通常是 1024）
        """
        return Config.VECTOR_DIMENSION

    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为向量

        这是 RAG 流程中的核心步骤：
        1. 接收文本输入
        2. 发送给 Ollama 的 embedding 模型
        3. 模型将文本编码为固定维度的数值向量
        4. 返回向量（一个浮点数列表）

        向量表示原理：
        - 计算机无法直接理解文字
        - embedding 模型将语义相似的文本映射到向量空间中相近的位置
        - "苹果是水果" 和 "苹果是一种水果" 的向量会很接近
        - "苹果是水果" 和 "汽车是交通工具" 的向量会距离较远

        Args:
            text: 输入文本

        Returns:
            文本的向量表示（浮点数列表）
        """
        try:
            # 调用 Ollama API 获取文本的向量表示
            # ollama.embed() 会调用 /api/embeddings 端点
            response = self._client.embeddings(
                model=self.embedding_model,
                prompt=text
            )

            embedding = response['embedding']

            print(f"[向量化] 文本长度: {len(text)} 字符")
            print(f"[向量化] 向量维度: {len(embedding)}")
            print(f"[向量化] 向量前5维: {embedding[:5]}")

            return embedding

        except Exception as e:
            error_msg = str(e)
            print(f"[向量化错误] {error_msg if error_msg else '未知错误'}", file=sys.stderr)
            # 返回零向量作为降级方案
            print(f"[向量化] 使用零向量作为降级方案")
            return [0.0] * self.get_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量将多个文本转换为向量

        比逐个调用 embed_text() 更高效：
        - 减少网络往返次数
        - 模型可以更好地利用 GPU 并行计算

        Args:
            texts: 文本列表

        Returns:
            向量列表（每个文本对应一个向量）
        """
        embeddings = []

        print(f"[批量向量化] 开始处理 {len(texts)} 个文本...")

        for i, text in enumerate(texts):
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)

                # 每处理 10 个文本打印一次进度
                if (i + 1) % 10 == 0:
                    print(f"[批量向量化] 已处理: {i + 1}/{len(texts)}")

            except Exception as e:
                print(f"[批量向量化错误] 第 {i + 1} 个文本失败: {str(e)}")
                # 使用零向量作为占位
                embeddings.append([0.0] * self.get_embedding_dimension())

        print(f"[批量向量化] 完成! 成功: {len(embeddings)}/{len(texts)}")

        return embeddings

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        发送对话请求，获取完整回答

        Args:
            messages: 对话历史，格式为 [{'role': 'user', 'content': '...'}, ...]
            system_prompt: 系统提示词（可选）
            temperature: 温度参数，控制随机性（0-1，越低越确定）
            max_tokens: 最大生成 token 数

        Returns:
            LLM 生成的回答文本
        """
        # 如果提供了系统提示，添加到消息开头
        if system_prompt:
            full_messages = [{'role': 'system', 'content': system_prompt}] + messages
        else:
            full_messages = messages

        try:
            print(f"[LLM 对话] 发送 {len(full_messages)} 条消息")
            print(f"[LLM 对话] 模型: {self.llm_model}")

            response = self._client.chat(
                model=self.llm_model,
                messages=full_messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )

            answer = response['message']['content']
            print(f"[LLM 对话] 回答长度: {len(answer)} 字符")

            return answer

        except Exception as e:
            error_msg = str(e)
            print(f"[LLM 对话错误] {error_msg if error_msg else '未知错误'}", file=sys.stderr)
            # 返回降级回答
            return "抱歉，模型服务暂时不可用，请稍后重试。如果问题持续存在，请检查 Ollama 服务是否正常运行。"

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Iterator[Dict]:
        """
        发送对话请求，以流式方式返回回答

        流式响应的优势：
        - 用户可以立即看到回答的每个字
        - 不需要等待完整回答才开始显示
        - 提供更好的用户体验

        Args:
            messages: 对话历史
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Yields:
            包含生成内容的字典 {'content': '...', 'done': bool}
        """
        # 如果提供了系统提示，添加到消息开头
        if system_prompt:
            full_messages = [{'role': 'system', 'content': system_prompt}] + messages
        else:
            full_messages = messages

        try:
            print(f"[LLM 流式对话] 开始生成...")
            print(f"[LLM 流式对话] 使用模型: {self.llm_model}")
            print(f"[LLM 流式对话] 消息数: {len(full_messages)}")

            stream = self._client.chat(
                model=self.llm_model,
                messages=full_messages,
                stream=True,  # 启用流式输出
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )

            chunk_count = 0
            full_content = []
            for chunk in stream:
                chunk_count += 1
                # 调试：打印第一个 chunk 的结构
                if chunk_count == 1:
                    print(f"[LLM 流式对话] 第一个 chunk 结构: {chunk}")
                
                # 兼容不同的 chunk 格式
                content = None
                done = chunk.get('done', False)
                
                # 尝试从 message.content 获取
                if 'message' in chunk:
                    content = chunk['message'].get('content', '')
                # 或者直接从 chunk 获取 content
                elif 'content' in chunk:
                    content = chunk.get('content', '')
                
                # 兼容 Ollama 的 aggregate 模式：done=True 时可能没有 content
                if content is None:
                    content = ''
                
                if content:
                    full_content.append(content)
                    yield {'message': {'content': content}, 'done': False}
                elif done and chunk_count > 1:
                    # 流式结束，如果之前没有发送任何内容，发送一个完成信号
                    break

            # 如果没有任何内容，尝试非流式方式作为备选
            if not full_content:
                print(f"[LLM 流式对话] 流式返回为空，尝试非流式方式...")
                try:
                    response = self._client.chat(
                        model=self.llm_model,
                        messages=full_messages,
                        stream=False,
                        options={
                            'temperature': temperature,
                            'num_predict': max_tokens,
                        }
                    )
                    content = response.get('message', {}).get('content', '')
                    if content:
                        print(f"[LLM 流式对话] 非流式返回内容: {content[:50]}...")
                        yield {'message': {'content': content}, 'done': False}
                except Exception as e:
                    print(f"[LLM 流式对话] 非流式备选也失败: {e}")

            print(f"[LLM 流式对话] 生成完成，共 {chunk_count} 个 chunk")
            yield {'message': {'content': ''}, 'done': True}

        except Exception as e:
            error_msg = str(e)
            print(f"[LLM 流式对话错误] {error_msg if error_msg else '未知错误'}", file=sys.stderr)
            # 返回错误消息
            yield {'message': {'content': f'抱歉，模型服务暂时不可用: {error_msg if error_msg else "未知错误"}'}, 'done': True}

    def classify_intent(self, query: str) -> Dict:
        """
        识别用户问题的意图

        Args:
            query: 用户输入的问题

        Returns:
            包含意图识别结果的字典
        """
        import json
        import re
        from app.config import Config

        try:
            print(f"[意图识别] 分析问题: {query[:50]}...")

            # 构建提示词
            prompt = Config.INTENT_USER_PROMPT_TEMPLATE.format(query=query)

            messages = [{'role': 'user', 'content': prompt}]

            response = self._client.chat(
                model=self.llm_model,
                messages=messages,
                options={
                    'temperature': 0.1,
                    'num_predict': 500,
                }
            )

            result_text = response['message']['content'].strip()
            print(f"[意图识别] 原始响应: {repr(result_text[:300])}")

            # 清理响应
            result_text = re.sub(r'^```json\s*', '', result_text, flags=re.MULTILINE)
            result_text = re.sub(r'^```\s*', '', result_text, flags=re.MULTILINE)
            result_text = re.sub(r'```\s*$', '', result_text, flags=re.MULTILINE)
            result_text = result_text.strip()

            # 尝试直接解析
            try:
                result = json.loads(result_text)
                print(f"[意图识别] 直接解析成功")
            except json.JSONDecodeError as e:
                print(f"[意图识别] 直接解析失败: {e}")
                # 提取 JSON
                match = re.search(r'\{[\s\S]*\}', result_text)
                if match:
                    json_str = match.group()
                    print(f"[意图识别] 提取JSON: {repr(json_str[:200])}")
                    try:
                        result = json.loads(json_str)
                        print(f"[意图识别] 提取解析成功")
                    except json.JSONDecodeError as e2:
                        print(f"[意图识别] 提取JSON解析失败: {e2}")
                        return {
                            'intent': 'KNOWLEDGE_QA',
                            'confidence': 0.5,
                            'reasoning': f'JSON解析失败: {str(e2)[:50]}',
                            'need_retrieval': True,
                            'suggested_rewrite': query
                        }
                else:
                    print(f"[意图识别] 未找到JSON对象")
                    return {
                        'intent': 'KNOWLEDGE_QA',
                        'confidence': 0.5,
                        'reasoning': '响应中未找到JSON对象',
                        'need_retrieval': True,
                        'suggested_rewrite': query
                    }

            print(f"[意图识别] 结果: intent={result.get('intent')}, confidence={result.get('confidence')}")
            return result

        except Exception as e:
            import traceback
            print(f"[意图识别错误] {str(e)}")
            print(f"[意图识别错误] {traceback.format_exc()}")
            return {
                'intent': 'KNOWLEDGE_QA',
                'confidence': 0.5,
                'reasoning': f'意图识别失败: {str(e)[:50]}',
                'need_retrieval': True,
                'suggested_rewrite': query
            }


# 全局客户端实例（延迟初始化）
_client_instance: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """
    获取 Ollama 客户端单例

    Returns:
        OllamaClient 实例
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = OllamaClient()
    return _client_instance


def demo_usage():
    """
    使用示例 - 演示 Ollama 客户端的用法
    """
    # 创建客户端
    client = OllamaClient()

    # 1. 检查连接
    print("=" * 50)
    print("1. 检查 Ollama 连接")
    print("=" * 50)
    status = client.check_connection()
    print(f"状态: {status}")

    # 2. 文本向量化
    print("\n" + "=" * 50)
    print("2. 文本向量化示例")
    print("=" * 50)

    texts = [
        "机器学习是人工智能的一个重要分支",
        "深度学习是机器学习的一个子领域",
        "今天天气真好，适合出去游玩"
    ]

    for text in texts:
        vector = client.embed_text(text)
        print(f"文本: {text[:30]}...")
        print(f"向量维度: {len(vector)}, 前5维: {vector[:5]}\n")

    # 3. 问答对话
    print("\n" + "=" * 50)
    print("3. 问答对话示例")
    print("=" * 50)

    messages = [
        {'role': 'user', 'content': '请简单介绍一下什么是机器学习？'}
    ]

    answer = client.chat(messages)
    print(f"回答: {answer}")


if __name__ == '__main__':
    demo_usage()
