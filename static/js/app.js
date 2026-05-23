/**
 * 本地 RAG 知识库问答系统 - 前端脚本
 */

// ============ 全局状态 ============
let selectedFile = null;
let conversationHistory = [];  // 多轮对话历史
let currentConversationId = null;  // 当前对话 ID
let conversationsList = [];  // 对话列表

// ============ DOM 元素 ============
document.addEventListener('DOMContentLoaded', () => {
    // 初始化
    initSidebar();
    initUploadArea();
    initUploadButton();
    initConfigInputs();
    initStatusCheck();
    loadConversations();  // 从服务端加载对话列表
    initKnowledgeBase();
    initQASection();
    initNewConversationBtn();
});

// ============ 新对话按钮 ============
function initNewConversationBtn() {
    const newConvBtn = document.getElementById('newConversationBtn');
    if (newConvBtn) {
        newConvBtn.addEventListener('click', async () => {
            // 创建新对话（服务端）
            await createNewConversation();

            // 清空对话容器
            const container = document.getElementById('conversationContainer');
            if (container) {
                container.innerHTML = '';
            }

            // 显示欢迎消息
            const welcomeMsg = document.getElementById('welcomeMessage');
            if (welcomeMsg) {
                welcomeMsg.style.display = 'flex';
            }

            // 切换到问答页面
            const chatTab = document.querySelector('[data-tab="chat"]');
            if (chatTab) {
                chatTab.click();
            }

            // 聚焦输入框
            const questionInput = document.getElementById('questionInput');
            if (questionInput) {
                questionInput.focus();
            }

            // 隐藏检索结果按钮
            const chunksToggleBtn = document.getElementById('chunksToggleBtn');
            if (chunksToggleBtn) {
                chunksToggleBtn.style.display = 'none';
            }

            // 关闭检索结果侧边栏
            closeChunksSidebar();

            // 隐藏流程指示器
            showQAFlow(false);
        });
    }
}

// ============ 侧边栏导航 ============
function initSidebar() {
    // 选择侧边栏中所有的导航项（包括底部设置）
    const navItems = document.querySelectorAll('.sidebar .nav-item');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();

            const tabName = item.getAttribute('data-tab');
            if (!tabName) return;

            // 切换导航项激活状态
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // 切换内容区
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });

            const targetTab = document.getElementById(`tab-${tabName}`);
            if (targetTab) {
                targetTab.classList.add('active');
                targetTab.style.display = 'block';
            }
        });
    });

    // 侧边栏展开/收缩
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');

    // 清除之前的收起状态缓存，确保默认展开
    localStorage.removeItem('sidebar_collapsed');

    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            // 保存状态
            localStorage.setItem('sidebar_collapsed', sidebar.classList.contains('collapsed'));
        });
    }
}

// ============ 上传区域 ============
function initUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    // 点击上传
    uploadArea.addEventListener('click', () => fileInput.click());

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectFile(e.target.files[0]);
        }
    });

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            selectFile(e.dataTransfer.files[0]);
        }
    });
}

function selectFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['md', 'txt'].includes(ext)) {
        alert('只支持 .md 和 .txt 格式的文件');
        return;
    }
    selectedFile = file;
    document.getElementById('uploadBtn').disabled = false;

    // 显示文件名
    const uploadArea = document.getElementById('uploadArea');
    const existingFilename = uploadArea.querySelector('.selected-filename');
    if (existingFilename) existingFilename.remove();

    const filename = document.createElement('p');
    filename.className = 'selected-filename';
    filename.textContent = `已选择: ${file.name} (${formatFileSize(file.size)})`;
    filename.style.color = '#10b981';
    filename.style.marginTop = '10px';
    uploadArea.appendChild(filename);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============ 上传按钮 ============
function initUploadButton() {
    document.getElementById('uploadBtn').addEventListener('click', uploadFiles);
}

async function uploadFiles() {
    if (!selectedFile) return;

    const uploadBtn = document.getElementById('uploadBtn');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressFilename = document.getElementById('progressFilename');
    const progressSteps = document.getElementById('progressSteps');

    // 禁用按钮，显示进度
    uploadBtn.disabled = true;
    uploadBtn.textContent = '上传中...';
    uploadProgress.style.display = 'block';
    progressFilename.textContent = selectedFile.name;

    // 清空并添加步骤
    progressSteps.innerHTML = '';
    const steps = ['解析文档', '文本分段', '向量化', '存储'];
    // Hybrid 模式下添加 BM25 索引步骤
    if (window.HYBRID_ENABLED) {
        steps.push('BM25索引');
    }
    steps.forEach((step, i) => {
        const div = document.createElement('div');
        div.className = 'progress-step';
        div.id = `step${i}`;
        div.textContent = step;
        progressSteps.appendChild(div);
    });

    // 构建 FormData
    const formData = new FormData();
    formData.append('file', selectedFile);

    // 获取配置参数
    const chunkSize = document.getElementById('chunkSize').value;
    const chunkOverlap = document.getElementById('chunkOverlap').value;

    if (chunkSize) {
        formData.append('chunk_size', chunkSize);
    }

    if (chunkOverlap) {
        formData.append('chunk_overlap', chunkOverlap);
    }

    try {
        // 更新步骤状态
        document.getElementById('step0').classList.add('active');

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // 更新所有步骤为完成状态
        for (let i = 0; i < steps.length; i++) {
            document.getElementById(`step${i}`).classList.remove('active');
            document.getElementById(`step${i}`).classList.add('done');
        }

        if (result.status === 'success') {
            document.getElementById('step0').textContent = `✓ ${steps[0]}`;
            document.getElementById('step1').textContent = `✓ ${steps[1]}`;
            document.getElementById('step2').textContent = `✓ ${steps[2]}`;
            document.getElementById('step3').textContent = `✓ ${steps[3]}`;

            // 显示成功信息
            progressFilename.textContent = `✓ ${result.filename}`;
            uploadBtn.textContent = '上传成功！';

            // 刷新知识库
            loadKnowledgeBase();

            // 重置
            setTimeout(() => {
                selectedFile = null;
                document.getElementById('fileInput').value = '';
                uploadProgress.style.display = 'none';
                uploadBtn.disabled = true;
                uploadBtn.textContent = '开始上传并构建知识库';

                const existingFilename = document.querySelector('.selected-filename');
                if (existingFilename) existingFilename.remove();
            }, 2000);
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        // 显示错误
        document.getElementById('step0').classList.add('error');
        progressFilename.textContent = `✗ ${error.message}`;
        uploadBtn.textContent = '上传失败，点击重试';
        uploadBtn.disabled = false;
    }
}

// ============ 配置输入 ============
function initConfigInputs() {
    // 配置变更时保存到 localStorage
    const inputs = ['chunkSize', 'chunkOverlap'];
    inputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            // 从 localStorage 恢复
            const saved = localStorage.getItem(`config_${id}`);
            if (saved !== null) {
                if (el.type === 'checkbox') {
                    el.checked = saved === 'true';
                } else {
                    el.value = saved;
                }
            }

            // 保存到 localStorage
            el.addEventListener('change', () => {
                if (el.type === 'checkbox') {
                    localStorage.setItem(`config_${id}`, el.checked);
                } else {
                    localStorage.setItem(`config_${id}`, el.value);
                }
            });
        }
    });
}

// ============ 状态检查 ============
function initStatusCheck() {
    checkStatus();
    // 每 30 秒检查一次状态
    setInterval(checkStatus, 30000);
}

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const statusGrid = document.getElementById('statusGrid');

        if (data.status === 'success') {
            statusGrid.innerHTML = `
                <div class="status-item ${data.ollama.status}">
                    <span class="status-label">Ollama</span>
                    <span class="status-value">${data.ollama.status === 'connected' ? '✓ 已连接' : '✗ 未连接'}</span>
                </div>
                <div class="status-item ${data.vector_db.status}">
                    <span class="status-label">向量数据库</span>
                    <span class="status-value">${data.vector_db.status === 'connected' ? '✓ 已连接' : '✗ 未连接'}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Embedding 模型</span>
                    <span class="status-value">${data.ollama.embedding_model}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">LLM 模型</span>
                    <span class="status-value">${data.ollama.llm_model}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">向量维度</span>
                    <span class="status-value">${data.vector_db.dimension}</span>
                </div>
                ${data.retrieval && data.retrieval.hybrid_enabled ? `
                <div class="status-item hybrid-enabled">
                    <span class="status-label">检索模式</span>
                    <span class="status-value">✓ Hybrid (BM25+Vector)</span>
                </div>
                <div class="status-item">
                    <span class="status-label">BM25/Vector 权重</span>
                    <span class="status-value">${(data.retrieval.bm25_weight * 100).toFixed(0)}% / ${((1 - data.retrieval.bm25_weight) * 100).toFixed(0)}%</span>
                </div>
                ` : ''}
            `;

            // 更新知识库统计
            updateKBStats(data.knowledge_base);

            // 存储 hybrid 状态
            if (data.retrieval) {
                window.HYBRID_ENABLED = data.retrieval.hybrid_enabled;
            }
        }
    } catch (error) {
        console.error('状态检查失败:', error);
    }
}

function updateKBStats(kbInfo) {
    document.getElementById('totalEntities').textContent = kbInfo.total_entities;
    document.getElementById('docCount').textContent = kbInfo.document_count;

    // 更新文档筛选器
    const docFilter = document.getElementById('docFilter');
    const currentValue = docFilter.value;
    docFilter.innerHTML = '<option value="">所有文档</option>';

    (kbInfo.documents || []).forEach(doc => {
        const option = document.createElement('option');
        option.value = doc;
        option.textContent = doc;
        docFilter.appendChild(option);
    });

    // 恢复之前的选择
    if (currentValue) {
        docFilter.value = currentValue;
    }
}

// ============ 知识库 ============
function initKnowledgeBase() {
    loadKnowledgeBase();

    document.getElementById('docFilter').addEventListener('change', loadKnowledgeBase);
    document.getElementById('refreshKbBtn').addEventListener('click', loadKnowledgeBase);
    document.getElementById('clearKbBtn').addEventListener('click', clearKnowledgeBase);
}

async function loadKnowledgeBase() {
    try {
        console.log('[前端] 开始加载知识库...');
        const response = await fetch('/api/knowledge-base');
        const data = await response.json();
        console.log('[前端] 知识库 API 返回:', data);

        if (data.status === 'success') {
            console.log('[前端] 文档数量:', data.document_count);
            console.log('[前端] 文档列表:', data.documents);
            console.log('[前端] 分段数量:', data.chunks?.length);
            displayKnowledgeBase(data.chunks || [], data.documents || []);
        } else {
            console.error('[前端] 加载失败:', data.message);
        }
    } catch (error) {
        console.error('[前端] 加载知识库失败:', error);
    }
}

function displayKnowledgeBase(chunks, documents) {
    const kbContent = document.getElementById('kbContent');
    const docFilter = document.getElementById('docFilter').value;

    console.log('[前端] displayKnowledgeBase 调用:', { chunks, documents, docFilter });

    // 检查第一个 chunk 的结构
    if (chunks.length > 0) {
        console.log('[前端] 第一个 chunk 的结构:', Object.keys(chunks[0]));
        console.log('[前端] 第一个 chunk:', chunks[0]);
    }

    if (!chunks || chunks.length === 0) {
        console.log('[前端] 知识库为空，显示空状态');
        kbContent.innerHTML = `<div class="kb-empty">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>
            <p>知识库为空，请先上传文档</p>
        </div>`;
        return;
    }

    kbContent.innerHTML = '';

    // 按文档分组
    const groupedByDoc = {};
    chunks.forEach(chunk => {
        const docName = chunk.doc_name || '未知文档';
        if (!groupedByDoc[docName]) {
            groupedByDoc[docName] = [];
        }
        groupedByDoc[docName].push(chunk);
    });

    console.log('[前端] 按文档分组:', Object.keys(groupedByDoc));

    // 如果只有一个文档，默认展开
    const singleDocMode = Object.keys(groupedByDoc).length === 1;

    Object.entries(groupedByDoc).forEach(([docName, docChunks]) => {
        // 过滤
        if (docFilter && docName !== docFilter) {
            return;
        }

        const isExpanded = singleDocMode || docFilter;
        console.log('[前端] 渲染文档:', docName, '分段数:', docChunks.length, 'isExpanded:', isExpanded);

        const docDiv = document.createElement('div');
        docDiv.className = 'kb-doc';
        docDiv.innerHTML = `
            <div class="kb-doc-header" onclick="toggleDoc(this)">
                <h4>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>
                    <span>${escapeHtml(docName)}</span>
                    <span style="color: #64748b; font-weight: normal;">(${docChunks.length} 个分段)</span>
                </h4>
                <span class="toggle-icon">${isExpanded ? '▲' : '▼'}</span>
            </div>
            <div class="kb-doc-body" style="display: ${isExpanded ? 'block' : 'none'};"></div>
        `;

        const body = docDiv.querySelector('.kb-doc-body');

        docChunks.forEach((chunk, index) => {
            const chunkDiv = document.createElement('div');
            chunkDiv.className = 'kb-chunk';
            const text = chunk.text || chunk || '';
            chunkDiv.innerHTML = `
                <div class="kb-chunk-header">
                    <span>分段 ${index + 1}</span>
                    <span class="chunk-expand-hint">点击展开</span>
                </div>
                <div class="kb-chunk-text" onclick="toggleChunkText(this)">${escapeHtml(text)}</div>
            `;
            body.appendChild(chunkDiv);
        });

        // 添加删除按钮
        const actionDiv = document.createElement('div');
        actionDiv.className = 'kb-doc-actions';
        actionDiv.innerHTML = `
            <button class="btn btn-danger" onclick="deleteDocument('${escapeHtml(docName)}')">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>
                删除此文档
            </button>
        `;
        body.appendChild(actionDiv);

        kbContent.appendChild(docDiv);
    });

    // 初始化 Lucide 图标
    lucide.createIcons();
}

function toggleDoc(header) {
    const doc = header.parentElement;
    const body = doc.querySelector('.kb-doc-body');
    const icon = doc.querySelector('.toggle-icon');

    if (body.style.display === 'none') {
        body.style.display = 'block';
        icon.textContent = '▲';
    } else {
        body.style.display = 'none';
        icon.textContent = '▼';
    }
}

async function deleteDocument(docName) {
    if (!confirm(`确定要删除文档「${docName}」吗？`)) {
        return;
    }

    try {
        const response = await fetch(`/api/knowledge-base/${encodeURIComponent(docName)}`, {
            method: 'DELETE'
        });
        const data = await response.json();

        if (data.status === 'success') {
            alert('文档已删除');
            loadKnowledgeBase();
            updateDocFilterAfterDelete();
        } else {
            alert('删除失败: ' + (data.message || '未知错误'));
        }
    } catch (error) {
        console.error('删除文档失败:', error);
        alert('删除失败: ' + error.message);
    }
}

async function clearKnowledgeBase() {
    if (!confirm('确定要清空整个知识库吗？此操作不可恢复！')) {
        return;
    }

    try {
        const response = await fetch('/api/knowledge-base/clear', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({})
        });
        const data = await response.json();

        if (data.status === 'success') {
            alert('知识库已清空');
            loadKnowledgeBase();
            document.getElementById('docFilter').innerHTML = '<option value="">所有文档</option>';
        } else {
            alert('清空失败: ' + (data.message || '未知错误'));
        }
    } catch (error) {
        console.error('清空知识库失败:', error);
        alert('清空失败: ' + error.message);
    }
}

function updateDocFilterAfterDelete() {
    const docFilter = document.getElementById('docFilter');
    const currentValue = docFilter.value;
    fetch('/api/knowledge-base')
        .then(r => r.json())
        .then(data => {
            if (data.documents) {
                docFilter.innerHTML = '<option value="">所有文档</option>';
                data.documents.forEach(doc => {
                    const opt = document.createElement('option');
                    opt.value = doc;
                    opt.textContent = doc;
                    docFilter.appendChild(opt);
                });
            }
        });
}

function expandChunk(chunk) {
    alert(`文档: ${chunk.doc_name}\n\n${chunk.text}`);
}

function toggleChunkText(element) {
    element.classList.toggle('expanded');
    // 查找提示文本（可能在父容器的 header 中）
    const header = element.previousElementSibling;
    if (header) {
        const hint = header.querySelector('.chunk-expand-hint');
        if (hint) {
            hint.textContent = element.classList.contains('expanded') ? '点击收起' : '点击展开';
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============ 问答 ============
function initQASection() {
    const questionInput = document.getElementById('questionInput');
    const askBtn = document.getElementById('askBtn');
    const clearConversationBtn = document.getElementById('clearConversationBtn');
    const chunksToggleBtn = document.getElementById('chunksToggleBtn');
    const chunksSidebar = document.getElementById('chunksSidebar');
    const chunksSidebarClose = document.getElementById('chunksSidebarClose');
    const chunksSidebarOverlay = document.getElementById('chunksSidebarOverlay');

    askBtn.addEventListener('click', askQuestion);
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });

    // 清空对话
    if (clearConversationBtn) {
        clearConversationBtn.addEventListener('click', () => {
            // 如果有当前对话，删除它
            if (currentConversationId) {
                deleteConversation(currentConversationId, { stopPropagation: () => {}, preventDefault: () => {} });
            }
            clearConversation();
        });
    }

    // 检索结果侧边栏开关
    if (chunksToggleBtn) {
        chunksToggleBtn.addEventListener('click', () => {
            openChunksSidebar();
        });
    }

    if (chunksSidebarClose) {
        chunksSidebarClose.addEventListener('click', () => {
            closeChunksSidebar();
        });
    }

    if (chunksSidebarOverlay) {
        chunksSidebarOverlay.addEventListener('click', () => {
            closeChunksSidebar();
        });
    }
}

function openChunksSidebar() {
    const chunksSidebar = document.getElementById('chunksSidebar');
    const chunksSidebarOverlay = document.getElementById('chunksSidebarOverlay');
    if (chunksSidebar) chunksSidebar.classList.add('open');
    if (chunksSidebarOverlay) chunksSidebarOverlay.classList.add('open');
}

function closeChunksSidebar() {
    const chunksSidebar = document.getElementById('chunksSidebar');
    const chunksSidebarOverlay = document.getElementById('chunksSidebarOverlay');
    if (chunksSidebar) chunksSidebar.classList.remove('open');
    if (chunksSidebarOverlay) chunksSidebarOverlay.classList.remove('open');
}

async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();

    if (!question) {
        alert('请输入问题');
        return;
    }

    const askBtn = document.getElementById('askBtn');

    // 禁用输入
    questionInput.disabled = true;
    askBtn.disabled = true;

    // 先显示用户消息
    const container = document.getElementById('conversationContainer');
    if (container) {
        const welcomeMsg = document.getElementById('welcomeMessage');
        if (welcomeMsg) welcomeMsg.style.display = 'none';

        const userDiv = document.createElement('div');
        userDiv.className = 'chat-bubble user-bubble';
        userDiv.textContent = question;
        container.appendChild(userDiv);

        // 创建助手消息占位
        const assistantDiv = document.createElement('div');
        assistantDiv.className = 'chat-bubble assistant-bubble';
        assistantDiv.textContent = '正在思考...';
        container.appendChild(assistantDiv);
    }

    // 显示流程
    showQAFlow(true);
    resetQAFlow();
    setQAFlowStep(0, '意图识别', 'pending');
    setQAFlowStep(1, '向量化', 'pending');
    setQAFlowStep(2, '知识库检索', 'pending');
    setQAFlowStep(3, '本地Rerank', 'pending');
    setQAFlowStep(4, '上下文构建', 'pending');
    setQAFlowStep(5, 'LLM 生成', 'pending');

    // 用于在流式事件处理中共享状态
    const streamState = { fullAnswer: '', question: question, conversationId: currentConversationId };

    try {
        // 构建请求数据 - 始终使用关联对话 API，让后端处理创建对话
        const requestData = {
            question: question,
            top_k: 5,
            stream: true,
            history: conversationHistory  // 发送对话历史用于 LLM 上下文
        };

        // 如果有当前对话 ID，添加到请求中
        if (currentConversationId) {
            requestData.conversation_id = currentConversationId;
        }

        const response = await fetch('/api/query-with-conversation', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(requestData)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.trim()) continue;

                try {
                    const data = JSON.parse(line);
                    await handleStreamEvent(data, streamState);
                } catch (e) {
                    console.error('Parse error:', e);
                }
            }
        }

        // 刷新对话列表（更新对话标题等）
        await loadConversations();

    } catch (error) {
        console.error('[前端] 请求失败:', error);
    } finally {
        // 恢复输入
        if (questionInput) {
            questionInput.disabled = false;
            questionInput.value = '';
        }
        if (askBtn) {
            askBtn.disabled = false;
            askBtn.textContent = '提问';
        }
        showQAFlow(false);
    }
}

async function handleStreamEvent(data, streamState) {
    switch (data.type) {
        case 'status':
            // 更新状态显示
            updateQAFlowStep(data.step, data.step_name, 'active', data.message);
            if (data.done) {
                updateQAFlowStep(data.step, data.step_name, 'done', data.message, data.detail);
            }
            break;

        case 'chitchat':
            // 闲聊回复
            streamState.fullAnswer = data.answer;

            // 更新对话容器中最后一条助手消息
            const chitchatContainer = document.getElementById('conversationContainer');
            if (chitchatContainer) {
                const lastBubble = chitchatContainer.querySelector('.chat-bubble:last-child');
                if (lastBubble && lastBubble.classList.contains('assistant-bubble')) {
                    lastBubble.textContent = data.answer;
                }
            }

            // 添加到对话历史
            conversationHistory.push({ role: 'assistant', content: data.answer });
            break;

        case 'chunks':
            // 显示检索到的分段
            displayRelevantChunks(data.data);
            // 显示检索结果按钮
            const chunksToggleBtn = document.getElementById('chunksToggleBtn');
            const chunksCount = document.getElementById('chunksCount');
            if (chunksToggleBtn) {
                chunksToggleBtn.style.display = 'flex';
                if (chunksCount && data.data) {
                    chunksCount.textContent = data.data.length;
                }
            }
            break;

        case 'content':
            // 流式内容 - 更新对话历史中的最后一条助手消息
            streamState.fullAnswer += data.content;
            // 更新对话容器中最后一条助手消息
            const contentContainer = document.getElementById('conversationContainer');
            if (contentContainer) {
                const lastBubble = contentContainer.querySelector('.chat-bubble:last-child');
                if (lastBubble && lastBubble.classList.contains('assistant-bubble')) {
                    lastBubble.textContent = streamState.fullAnswer + '▌';  // 添加闪烁光标
                }
            }

            // 如果服务端返回了 conversation_id，更新当前对话 ID
            if (data.conversation_id && !streamState.conversationId) {
                streamState.conversationId = data.conversation_id;
                currentConversationId = data.conversation_id;
            }
            break;

        case 'done':
            // 完成 - 使用累积的完整回答
            const finalAnswer = data.answer || streamState.fullAnswer;
            streamState.fullAnswer = finalAnswer;

            // 如果服务端返回了 conversation_id，更新当前对话 ID
            if (data.conversation_id) {
                streamState.conversationId = data.conversation_id;
                currentConversationId = data.conversation_id;
            }

            // 更新对话容器中最后一条助手消息（移除光标）
            const doneContainer = document.getElementById('conversationContainer');
            if (doneContainer) {
                const lastBubble = doneContainer.querySelector('.chat-bubble:last-child');
                if (lastBubble && lastBubble.classList.contains('assistant-bubble')) {
                    lastBubble.textContent = finalAnswer;
                }
            }

            // 添加助手回复到对话历史（如果还没有添加）
            if (conversationHistory.length % 2 !== 0 ||
                conversationHistory[conversationHistory.length - 1]?.role !== 'assistant') {
                conversationHistory.push({ role: 'assistant', content: finalAnswer });
            }

            // 刷新对话列表
            await loadConversations();
            setQAFlowStep(5, 'LLM 生成', 'done', '回答生成完成');
            break;

        case 'error':
            console.error('[前端] 流式请求错误:', data.message);
            break;
    }
}

function resetQAFlow() {
    for (let i = 0; i <= 5; i++) {
        const step = document.getElementById(`qaStep${i}`);
        if (step) {
            step.className = 'qa-step';
        }
    }
}

function setQAFlowStep(step, name, status, message) {
    const stepEl = document.getElementById(`qaStep${step}`);
    if (stepEl) {
        stepEl.className = `qa-step ${status}`;
        const textEl = stepEl.querySelector('.step-text');
        if (textEl) {
            textEl.textContent = message || name;
        }
    }
}

function updateQAFlowStep(step, name, status, message, detail) {
    const stepEl = document.getElementById(`qaStep${step}`);
    if (stepEl) {
        stepEl.className = `qa-step ${status}`;
        const textEl = stepEl.querySelector('.step-text');
        if (textEl) {
            textEl.textContent = message || name;
        }
    }
}

function showQAFlow(show) {
    document.getElementById('qaFlow').style.display = show ? 'flex' : 'none';
}

function displayRelevantChunks(chunks) {
    const container = document.getElementById('chunksList');
    if (!container) return;

    container.innerHTML = '';

    if (!chunks || chunks.length === 0) {
        container.innerHTML = '<p class="no-results">未找到相关分段</p>';
        return;
    }

    chunks.forEach((chunk, i) => {
        const div = document.createElement('div');
        div.className = 'chunk-item';
        // 兼容 hybrid score 和普通 score
        const score = chunk.hybrid_score !== undefined ? chunk.hybrid_score : chunk.score;
        const retrievalType = chunk.retrieval_type || 'vector';

        // 判断是否使用了 rerank
        const hasRerank = chunk.rerank_applied && chunk.rerank_score !== undefined;
        const rerankScore = chunk.rerank_score || 0;

        // 构建类型标签
        let typeLabel = 'Vector';
        let scoreLabel = '相似度';
        if (retrievalType.includes('hybrid')) {
            typeLabel = 'Hybrid';
            scoreLabel = 'Hybrid分数';
        }
        if (hasRerank) {
            typeLabel = 'Rerank';
            scoreLabel = 'Rerank分数';
        }

        // 构建分数详情 HTML
        let scoreDetails = '';
        if (hasRerank && chunk.rerank_details) {
            const details = chunk.rerank_details;
            scoreDetails = `
                <div class="rerank-details">
                    <span class="rerank-badge">Rerank 精排</span>
                    <div class="rerank-scores">
                        <span class="score-item" title="向量相似度">向量: ${(details.vector_score * 100).toFixed(1)}%</span>
                        <span class="score-item" title="关键词覆盖率">关键词: ${(details.keyword_score * 100).toFixed(1)}%</span>
                        <span class="score-item" title="长度适配度">长度: ${(details.length_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="rerank-total">综合: <strong>${(rerankScore * 100).toFixed(1)}%</strong></div>
                </div>
            `;
        }

        div.innerHTML = `
            <div class="chunk-header">
                <span class="chunk-index">#${i + 1}</span>
                <span class="chunk-score">${scoreLabel}: ${(score * 100).toFixed(1)}%</span>
                <span class="chunk-type">${typeLabel}</span>
                <span class="chunk-doc">${chunk.doc_name}</span>
                <span class="chunk-expand-hint">点击展开</span>
            </div>
            ${scoreDetails}
            <div class="chunk-text" onclick="toggleChunkText(this)">${escapeHtml(chunk.text)}</div>
        `;
        container.appendChild(div);
    });
}

// ============ 对话管理 ============

/**
 * 从服务端加载对话列表
 */
async function loadConversations() {
    try {
        const response = await fetch('/api/conversations?limit=50');
        const data = await response.json();

        if (data.status === 'success') {
            conversationsList = data.conversations;
            updateSidebarConversations();
        }
    } catch (error) {
        console.error('[前端] 加载对话列表失败:', error);
    }
}

/**
 * 创建新对话
 */
async function createNewConversation() {
    try {
        const response = await fetch('/api/conversations', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({})
        });
        const data = await response.json();

        if (data.status === 'success') {
            currentConversationId = data.conversation_id;
            conversationHistory = [];
            displayConversationHistory();

            // 刷新对话列表
            await loadConversations();

            return data.conversation_id;
        }
    } catch (error) {
        console.error('[前端] 创建对话失败:', error);
    }
    return null;
}

/**
 * 加载指定对话
 */
async function loadConversation(conversationId) {
    console.log('[前端] 开始加载对话:', conversationId);

    // 显示加载提示
    const container = document.getElementById('conversationContainer');
    const welcomeMessage = document.getElementById('welcomeMessage');
    if (container) {
        container.innerHTML = '<div class="loading-conversation"><div class="loading-spinner"></div><span>正在加载对话...</span></div>';
    }

    try {
        const response = await fetch(`/api/conversations/${conversationId}`);
        const data = await response.json();

        console.log('[前端] API 返回:', data);

        if (data.status === 'success' && data.conversation) {
            currentConversationId = conversationId;

            // 提取消息
            const messages = data.conversation.messages || [];
            conversationHistory = messages.map(msg => ({
                role: msg.role,
                content: msg.content
            }));

            console.log('[前端] 设置 conversationHistory:', conversationHistory);
            console.log('[前端] 消息数量:', conversationHistory.length);

            displayConversationHistory();

            // 切换到问答页面
            const chatTab = document.querySelector('[data-tab="chat"]');
            if (chatTab) {
                chatTab.click();
            }

            // 更新侧边栏选中状态
            updateSidebarConversations();
        } else {
            console.error('[前端] 加载对话失败:', data.message);
            alert('加载对话失败: ' + (data.message || '未知错误'));
        }
    } catch (error) {
        console.error('[前端] 加载对话失败:', error);
        alert('加载对话失败: ' + error.message);
    }
}

/**
 * 删除对话
 */
async function deleteConversation(conversationId, event) {
    event.stopPropagation();
    event.preventDefault();

    if (!confirm('确定要删除这个对话吗？')) {
        return;
    }

    try {
        const response = await fetch(`/api/conversations/${conversationId}`, {
            method: 'DELETE'
        });
        const data = await response.json();

        if (data.status === 'success') {
            // 如果删除的是当前对话，清空界面
            if (currentConversationId === conversationId) {
                currentConversationId = null;
                conversationHistory = [];
                displayConversationHistory();
            }

            // 刷新对话列表
            await loadConversations();
        }
    } catch (error) {
        console.error('[前端] 删除对话失败:', error);
    }
}

/**
 * 更新侧边栏对话列表（从服务端数据）
 */
function updateSidebarConversations() {
    const sidebarList = document.getElementById('sidebarConversationList');
    if (!sidebarList) return;

    const messageIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
    const xIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';
    const inboxIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg>';

    if (conversationsList && conversationsList.length > 0) {
        sidebarList.innerHTML = '';
        conversationsList.forEach((conv) => {
            const div = document.createElement('button');
            div.className = 'conversation-item' + (conv.id === currentConversationId ? ' active' : '');
            div.title = conv.title;
            div.innerHTML = `
                ${messageIcon}
                <span class="conv-title">${escapeHtml(conv.title)}</span>
                <span class="conv-delete" data-id="${conv.id}">
                    ${xIcon}
                </span>
            `;
            div.addEventListener('click', (e) => {
                console.log('[前端] 点击对话:', conv.id, conv.title);
                loadConversation(conv.id);
            });

            // 删除按钮事件
            const deleteBtn = div.querySelector('.conv-delete');
            deleteBtn.addEventListener('click', (e) => {
                console.log('[前端] 点击删除:', conv.id);
                deleteConversation(conv.id, e);
            });

            sidebarList.appendChild(div);
        });
    } else {
        // 如果没有对话
        sidebarList.innerHTML = `
            <div class="conversation-empty">
                ${inboxIcon}
                <span>暂无对话记录</span>
            </div>
        `;
    }
}

function saveHistory() {
    // 历史已保存到服务端，无需额外操作
    // 刷新对话列表
    loadConversations();
}

function addToHistory(question, answer) {
    // 移除欢迎消息
    const welcomeMsg = document.getElementById('welcomeMessage');
    if (welcomeMsg) {
        welcomeMsg.style.display = 'none';
    }

    // 显示对话气泡
    displayConversationHistory();

    // 保存到服务端
    saveHistory();
}

/**
 * 清空当前对话
 */
function clearConversation() {
    currentConversationId = null;
    conversationHistory = [];
    displayConversationHistory();

    // 隐藏检索结果按钮
    const chunksToggleBtn = document.getElementById('chunksToggleBtn');
    if (chunksToggleBtn) {
        chunksToggleBtn.style.display = 'none';
    }

    // 关闭侧边栏
    closeChunksSidebar();

    // 清空相关分段
    const chunksList = document.getElementById('chunksList');
    if (chunksList) {
        chunksList.innerHTML = '';
    }

    console.log('[前端] 对话已清空');
}

// ============ 多轮对话 ============

/**
 * 显示对话历史记录
 * 在 QA 结果区域上方显示当前对话的上下文
 */
function displayConversationHistory() {
    const container = document.getElementById('conversationContainer');
    const welcomeMessage = document.getElementById('welcomeMessage');
    if (!container) {
        console.log('[前端] displayConversationHistory: container 不存在');
        return;
    }

    console.log('[前端] displayConversationHistory: history长度=', conversationHistory.length);

    // 如果没有历史，显示欢迎消息
    if (conversationHistory.length === 0) {
        container.innerHTML = '';
        if (welcomeMessage) welcomeMessage.style.display = 'flex';
        return;
    }

    // 有对话时隐藏欢迎消息
    if (welcomeMessage) welcomeMessage.style.display = 'none';

    // 多轮对话时显示简洁的对话气泡
    container.innerHTML = '';

    // 遍历对话历史，以聊天气泡形式展示
    for (let i = 0; i < conversationHistory.length; i += 2) {
        const userMsg = conversationHistory[i];
        const assistantMsg = conversationHistory[i + 1];

        // 用户消息
        if (userMsg) {
            const userDiv = document.createElement('div');
            userDiv.className = 'chat-bubble user-bubble';
            userDiv.textContent = userMsg.content;
            container.appendChild(userDiv);
        }

        // 助手消息
        if (assistantMsg) {
            const assistantDiv = document.createElement('div');
            assistantDiv.className = 'chat-bubble assistant-bubble';
            assistantDiv.textContent = assistantMsg.content;
            container.appendChild(assistantDiv);
        }
    }

    console.log('[前端] displayConversationHistory: 完成，container子元素数量=', container.children.length);
}

/**
 * 更新最后一条助手消息（用于流式输出）
 */
function updateLastAssistantMessage(content) {
    const container = document.getElementById('conversationContainer');
    if (!container) return;

    const lastBubble = container.querySelector('.chat-bubble:last-child');
    if (lastBubble && lastBubble.classList.contains('assistant-bubble')) {
        lastBubble.textContent = content + '▌';  // 添加闪烁光标
    }
}

