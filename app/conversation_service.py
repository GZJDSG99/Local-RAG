"""
对话管理服务
============

提供对话历史的持久化存储和管理功能。
支持创建对话、查询对话、追加消息、删除对话等操作。
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conversations.db')


def get_db_path():
    """获取数据库路径"""
    return DB_PATH


@contextmanager
def get_db_connection():
    """获取数据库连接的上下文管理器"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_conversation_db():
    """
    初始化对话数据库

    创建 conversations 表和 messages 表：
    - conversations: 存储对话会话（标题、创建时间、更新时间）
    - messages: 存储每条消息（属于某个对话）
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 创建对话会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT '新对话',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0
            )
        ''')

        # 创建消息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        ''')

        # 创建索引以加速查询
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
            ON messages(conversation_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
            ON conversations(updated_at DESC)
        ''')

        print(f"[ConversationDB] 数据库初始化完成: {DB_PATH}")


class ConversationService:
    """对话管理服务类"""

    def __init__(self):
        """初始化对话服务"""
        self._ensure_db_init()

    def _ensure_db_init(self):
        """确保数据库已初始化"""
        if not os.path.exists(DB_PATH):
            init_conversation_db()

    def create_conversation(self, title: str = None, first_message: str = None) -> int:
        """
        创建新对话会话

        Args:
            title: 对话标题，默认使用第一条消息的前20个字符
            first_message: 第一条消息内容，用于生成标题

        Returns:
            新对话的 ID
        """
        # 自动生成标题
        if not title:
            if first_message:
                title = first_message[:20] + ('...' if len(first_message) > 20 else '')
            else:
                title = '新对话'

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO conversations (title) VALUES (?)',
                (title,)
            )
            conversation_id = cursor.lastrowid
            print(f"[ConversationService] 创建新对话: ID={conversation_id}, title={title}")
            return conversation_id

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Dict = None
    ) -> int:
        """
        向对话添加消息

        Args:
            conversation_id: 对话 ID
            role: 角色 ('user' | 'assistant' | 'system')
            content: 消息内容
            metadata: 额外元数据（可选）

        Returns:
            新消息的 ID
        """
        if role not in ('user', 'assistant', 'system'):
            raise ValueError(f"Invalid role: {role}")

        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)',
                (conversation_id, role, content, metadata_json)
            )
            message_id = cursor.lastrowid

            # 更新对话的更新时间
            cursor.execute(
                '''
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP,
                    message_count = message_count + 1
                WHERE id = ?
                ''',
                (conversation_id,)
            )

            print(f"[ConversationService] 添加消息: conversation_id={conversation_id}, role={role}, message_id={message_id}")
            return message_id

    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """
        获取对话详情

        Args:
            conversation_id: 对话 ID

        Returns:
            对话信息，包含所有消息
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 获取对话信息
            cursor.execute(
                'SELECT * FROM conversations WHERE id = ?',
                (conversation_id,)
            )
            conv_row = cursor.fetchone()

            if not conv_row:
                return None

            # 获取所有消息
            cursor.execute(
                'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
                (conversation_id,)
            )
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'id': row['id'],
                    'role': row['role'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata'] or '{}'),
                    'created_at': row['created_at']
                })

            return {
                'id': conv_row['id'],
                'title': conv_row['title'],
                'created_at': conv_row['created_at'],
                'updated_at': conv_row['updated_at'],
                'message_count': conv_row['message_count'],
                'messages': messages
            }

    def get_conversation_messages(self, conversation_id: int) -> List[Dict]:
        """
        获取对话的所有消息（简洁格式，用于发送给 LLM）

        Args:
            conversation_id: 对话 ID

        Returns:
            消息列表，格式为 [{role: 'user', content: '...'}, ...]
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in conversation['messages']
        ]

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        获取对话列表

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            对话列表（不含消息内容）
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT id, title, created_at, updated_at, message_count
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                ''',
                (limit, offset)
            )

            return [
                {
                    'id': row['id'],
                    'title': row['title'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'message_count': row['message_count']
                }
                for row in cursor.fetchall()
            ]

    def update_conversation_title(self, conversation_id: int, title: str) -> bool:
        """
        更新对话标题

        Args:
            conversation_id: 对话 ID
            title: 新标题

        Returns:
            是否更新成功
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (title, conversation_id)
            )
            return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: int) -> bool:
        """
        删除对话及其所有消息

        Args:
            conversation_id: 对话 ID

        Returns:
            是否删除成功
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 先删除消息
            cursor.execute(
                'DELETE FROM messages WHERE conversation_id = ?',
                (conversation_id,)
            )

            # 再删除对话
            cursor.execute(
                'DELETE FROM conversations WHERE id = ?',
                (conversation_id,)
            )

            deleted = cursor.rowcount > 0
            if deleted:
                print(f"[ConversationService] 删除对话: ID={conversation_id}")
            return deleted

    def search_conversations(self, keyword: str, limit: int = 20) -> List[Dict]:
        """
        搜索对话（按标题和消息内容搜索）

        Args:
            keyword: 搜索关键词
            limit: 返回数量限制

        Returns:
            匹配的对话列表
        """
        search_pattern = f'%{keyword}%'

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 搜索标题或消息内容包含关键词的对话
            cursor.execute(
                '''
                SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at, c.message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.title LIKE ? OR m.content LIKE ?
                ORDER BY c.updated_at DESC
                LIMIT ?
                ''',
                (search_pattern, search_pattern, limit)
            )

            return [
                {
                    'id': row['id'],
                    'title': row['title'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'message_count': row['message_count']
                }
                for row in cursor.fetchall()
            ]

    def get_conversation_count(self) -> int:
        """
        获取对话总数

        Returns:
            对话数量
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM conversations')
            return cursor.fetchone()['count']


# 全局单例
_conversation_service = None


def get_conversation_service() -> ConversationService:
    """获取对话服务的单例实例"""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service
