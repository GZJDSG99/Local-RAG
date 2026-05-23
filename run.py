"""
Flask 应用主入口
================

启动本地 RAG 知识库问答系统的 Web 服务。

运行方式:
    python run.py

访问地址:
    http://localhost:5000
"""

import os
import sys

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.routes import api_bp


def create_app():
    """
    创建 Flask 应用实例

    Returns:
        Flask 应用对象
    """
    # 获取项目根目录的绝对路径（run.py 在根目录下）
    base_dir = os.path.dirname(os.path.abspath(__file__))

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, 'templates'),
        static_folder=os.path.join(base_dir, 'static')
    )

    # 配置
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

    # 启用 CORS，允许跨域请求
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # 注册 Blueprint
    app.register_blueprint(api_bp)

    # 确保上传目录存在
    upload_folder = os.path.join(base_dir, Config.UPLOAD_FOLDER)
    os.makedirs(upload_folder, exist_ok=True)

    return app


def main():
    """
    主函数 - 启动应用
    """
    print("\n" + "=" * 60)
    print("本地 RAG 知识库问答系统")
    print("=" * 60)
    print()
    print("技术栈:")
    print("  - Ollama: 本地 LLM 推理服务")
    print(f"    - 向量模型: {Config.EMBEDDING_MODEL}")
    print(f"    - 问答模型: {Config.LLM_MODEL}")
    print("  - Milvus: 开源向量数据库")
    print(f"    - 集合: {Config.COLLECTION_NAME}")
    print(f"    - 向量维度: {Config.VECTOR_DIMENSION}")
    print()
    print("配置:")
    print(f"  - 分段大小: {Config.DEFAULT_CHUNK_SIZE} 字符")
    print(f"  - 分段重叠: {Config.DEFAULT_CHUNK_OVERLAP} 字符")
    print(f"  - 检索数量: {Config.TOP_K}")
    print()
    print("访问地址: http://localhost:5000")
    print("=" * 60)
    print()

    # 创建应用
    app = create_app()

    # 启动服务
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True  # 启用多线程支持
    )


if __name__ == '__main__':
    main()
