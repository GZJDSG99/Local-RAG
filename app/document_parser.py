"""
文档解析模块
============

负责读取和解析 .md 和 .txt 格式的文档文件。
支持 Markdown 格式的文档解析。

教学要点:
- 文件读取：使用 Python 内置函数读取文本文件
- 编码处理：自动检测文件编码，避免乱码
- Markdown 解析：使用 markdown 库将 Markdown 转换为纯文本
  （因为向量模型更适合处理纯文本）
"""

import os
import chardet
import markdown
from typing import Optional
from werkzeug.datastructures import FileStorage


class DocumentParser:
    """
    文档解析器

    功能：
    - 检测文件编码
    - 读取文本内容
    - 将 Markdown 转换为纯文本
    """

    @staticmethod
    def detect_encoding(file_content: bytes) -> str:
        """
        自动检测文件编码

        为什么需要编码检测？
        - 文本文件可以有不同的字符编码（UTF-8、GBK、GB2312 等）
        - 如果用错误的编码读取文件，会出现乱码
        - chardet 库可以根据字节内容自动推断编码

        Args:
            file_content: 文件的原始字节内容

        Returns:
            编码名称字符串（如 'utf-8', 'gbk' 等）
        """
        # chardet.detect() 返回一个字典，包含编码信息和置信度
        result = chardet.detect(file_content)
        encoding = result['encoding']
        confidence = result['confidence']

        print(f"[编码检测] 检测到编码: {encoding}, 置信度: {confidence:.2%}")

        # 如果置信度较低，或者检测到的编码不合适，使用 UTF-8
        if confidence < 0.7 or encoding is None:
            return 'utf-8'

        # 统一转换为小写
        return encoding.lower()

    @staticmethod
    def read_file(file: FileStorage) -> tuple[str, str]:
        """
        读取文件内容

        Args:
            file: Flask 上传的文件对象

        Returns:
            (文件内容, 文件扩展名) 元组
        """
        # 获取文件扩展名（小写）
        filename = file.filename or 'unknown'
        extension = os.path.splitext(filename)[1].lower().lstrip('.')

        # 读取文件原始字节
        raw_content = file.read()

        # 检测编码
        encoding = DocumentParser.detect_encoding(raw_content)

        # 解码为字符串
        try:
            content = raw_content.decode(encoding)
        except UnicodeDecodeError:
            # 如果指定编码失败，尝试 UTF-8
            content = raw_content.decode('utf-8', errors='ignore')

        return content, extension

    @staticmethod
    def parse_markdown(md_content: str) -> str:
        """
        将 Markdown 格式转换为纯文本

        为什么需要转换？
        - Markdown 包含格式符号（#, *, -, > 等）
        - 这些符号对语义理解没有帮助，反而可能干扰向量模型
        - 转换为纯文本后，向量模型能更好地理解内容

        转换规则：
        - # 标题 → 保留标题文字
        - **粗体** → 粗体文字
        - - 列表 → 列表项文字
        - [链接](url) → 链接文字
        - 代码块 → 保留代码内容
        - 等等...

        Args:
            md_content: Markdown 格式的文本

        Returns:
            纯文本格式的内容
        """
        # 使用 markdown 库将 MD 转换为 HTML
        html = markdown.markdown(
            md_content,
            extensions=[
                'tables',      # 支持表格
                'fenced_code', # 支持代码块
                'nl2br',       # 换行符转换为 <br>
            ]
        )

        # 简单清理 HTML 标签，保留文本
        # 这是一个简化版本，生产环境可以使用更完整的 HTML 解析库
        import re
        text = re.sub(r'<br\s*/?>', '\n', html)  # <br> 转换为换行
        text = re.sub(r'</p>', '\n\n', text)     # </p> 转换为双换行
        text = re.sub(r'</div>', '\n', text)
        text = re.sub(r'<[^>]+>', '', text)      # 移除其他 HTML 标签
        text = re.sub(r'\n{3,}', '\n\n', text)   # 多个换行压缩为两个

        return text.strip()

    @staticmethod
    def parse_document(file: FileStorage) -> str:
        """
        解析文档，返回纯文本内容

        这是主入口函数，封装了解析流程：
        1. 读取文件 → 2. 检测编码 → 3. 转换为纯文本

        Args:
            file: Flask 上传的文件对象

        Returns:
            纯文本格式的文档内容
        """
        content, extension = DocumentParser.read_file(file)

        # 根据文件类型决定是否需要转换
        if extension == 'md':
            print(f"[文档解析] 解析 Markdown 文件: {file.filename}")
            content = DocumentParser.parse_markdown(content)
        elif extension == 'txt':
            print(f"[文档解析] 解析文本文件: {file.filename}")
            # .txt 文件已经是纯文本，不需要转换
            pass
        else:
            raise ValueError(f"不支持的文件格式: .{extension}")

        return content


def demo_usage():
    """
    使用示例 - 演示文档解析模块的用法
    """
    # 假设我们有一个文件对象
    # file = request.files['file']
    #
    # # 解析文档
    # content = DocumentParser.parse_document(file)
    # print(f"解析结果 (前200字符): {content[:200]}")
    pass
