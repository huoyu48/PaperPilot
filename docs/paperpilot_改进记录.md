## PaperPilot 问题修复记录

### 问题 1：上传论文后系统无法检索

**现象**：用户上传 PDF 后问"介绍下这篇论文"，系统完全忽略上传内容，只通过外部工具搜索，生成了一份与论文无关的通用报告。

**根因**：文档上传后正确进入了 ChromaDB 向量库，但 researcher 节点只调用了外部工具（arxiv、github、web_search、jina_reader），从未查询本地向量库。RAG 管道写好了却没接入研究流程。

**修复**：

1. **researcher.py** — 新增 `_search_local_docs()` 函数查询 ChromaDB，支持 `local_docs` 工具类型。同时加入"兜底自动检索"逻辑：即使 planner 没有分配 local_docs 子查询，researcher 也会用主查询自动检索向量库，有结果就注入。

```python
# researcher.py — 兜底自动检索
has_local_docs_sq = any(sq["tool"] == "local_docs" for sq in pending)
if not has_local_docs_sq:
    local_results = _search_local_docs(state["query"], top_k=3)
    if local_results:
        for item in local_results:
            all_results.append(ResearchResult(
                sub_query_id="rag_auto", tool="local_docs", ...
            ))
```

2. **planner.py** — system prompt 新增 `local_docs` 工具选项，并加入规则：当用户提到"上传的论文"、"这篇论文"等关键词时，必须生成至少一个 `local_docs` 类型的子查询。

**涉及文件**：`src/agent/researcher.py`, `src/agent/planner.py`

---

### 问题 2：报告生成后页面自动跳到底部

**现象**：报告很长，生成完毕后页面直接滚到最底部（"研究完成"消息处），用户想从头阅读还得手动翻上去。

**根因**：`addMessage()` 函数每次添加消息都会执行 `chat.scrollTop = chat.scrollHeight`，把页面拉到底部。`handleWSEvent` 末尾也有同样的滚动。报告渲染后立刻被拖到底部。

**修复**：

1. `addMessage()` 增加 `scroll` 参数（默认 true），返回创建的 DOM 元素
2. `showReport()` 调用 `addMessage(scroll=false)` 后用 `el.scrollIntoView({ behavior: 'smooth', block: 'start' })` 滚到报告顶部
3. `handleWSEvent` 末尾对 `done` 事件跳过滚动

```javascript
function addMessage(role, content, scroll = true) {
  // ... 创建元素 ...
  if (scroll) chat.scrollTop = chat.scrollHeight;
  return div;
}

function showReport(md) {
  // ... 渲染 markdown ...
  const el = addMessage('assistant', html, false);
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
```

**涉及文件**：`src/frontend/index.html`

---

### 问题 3：追问时丢失上下文（"要中文版"被当成新研究）

**现象**：用户先研究了知识追踪，得到英文报告。追问"我要中文版的介绍"，系统完全忘记了之前的研究，重新发起了 5 个全新的 web_search 子查询，生成了一份全新的报告。

**根因**：两个层面。

前端：虽然 `currentSessionId` 正确传递了 session_id，但没有发送对话历史。后端只拿到了原始查询和前次报告的 800 字符摘要，不够 planner 理解"中文版"指的是什么。

后端：旧的 follow-up 逻辑把 `effective_query` 拼成 `"Previous research question: ... Previous report summary: {report[:800]} Follow-up question: ..."`，planner 看到的上下文太少，无法判断这是一个语言转换请求。

**修复**：

1. **前端 conversationHistory 数组** — 追踪完整的会话历史（每次用户提问 + 每次完整报告），格式为 `[{role: 'user', content}, {role: 'assistant', report}]`。每次发请求时作为 `conversation_history` 字段传给后端。

2. **后端接收并格式化** — 将前端传来的数组转成 `[User]: ... \n\n [Assistant report]: ...` 格式的字符串，存入 AgentState 的 `conversation_history` 字段。

3. **planner.py 增强** — 新增 "Follow-up Questions" 指令段，识别三种追问类型：
   - 语言/格式转换（"中文版"、"translate"）→ 只生成 1 个子查询，writer 负责翻译
   - 深入某个方面 → 聚焦该方面生成子查询
   - 新的相关问题 → 正常生成子查询

4. **writer.py 增强** — 新增 "Follow-up / Conversation Context" 指令段，收到会话历史时：如果是语言转换请求，直接翻译已有报告内容，而非重新研究。

5. **AgentState** — 新增 `conversation_history: str` 字段，在 planner 和 writer 的 prompt 中注入。

6. **loadSession() 重建历史** — 点击历史记录加载时，从 session 数据重建 conversationHistory 数组，确保追问功能在加载历史后也能正常工作。

7. **newResearch() 清空历史** — 新建研究时清空 conversationHistory。

**涉及文件**：`src/agent/state.py`, `src/agent/planner.py`, `src/agent/writer.py`, `src/backend/websocket_handler.py`, `src/frontend/index.html`

---

### 问题 4：上下文无限增长，需要截断机制

**现象**：每轮追问都把完整历史（包括完整报告文本）发给 LLM。一份报告约 2000-3000 词 ≈ 3000-5000 tokens，3 轮追问后历史约 15-25K tokens，5 轮以上可能超出上下文窗口。

**模型限制**：DeepSeek-Chat 上下文 128K tokens，输出上限 4096 tokens。

**初版修复（被否定）**：简单截断 — 超过 48K 字符时丢弃最早的部分，只保留最近内容。

**用户反馈**："你别自动丢弃啊，你总结下前文不行吗？"

**最终修复**：LLM 摘要压缩。

```python
async def _summarize_history(history_str: str) -> str:
    if len(history_str) <= MAX_HISTORY_CHARS:  # 48K chars ≈ 12K tokens
        return history_str

    # 从中间找一个干净的对话边界
    mid = len(history_str) // 2
    cut = history_str.find("\n\n[User]:", mid)

    old_part = history_str[:cut]      # 旧对话 → 交给 LLM 总结
    recent_part = history_str[cut:]   # 最近对话 → 保留完整原文

    resp = await asyncio.to_thread(
        llm.invoke,
        [SystemMessage(content="请用中文简洁总结以下对话历史..."), ("human", old_part)],
    )

    return f"[历史对话摘要]\n{summary}\n\n---\n\n[最近对话]\n{recent_part}"
```

**流程**：历史 < 48K 字符 → 原样传递；超过 → LLM 把旧对话压缩成 ≤500 字的中文摘要，最近的对话保留完整原文。planner/writer 看到的是摘要 + 近期原文，既有全局视野又不丢关键信息。

**涉及文件**：`src/backend/websocket_handler.py`

---

### 问题 5：报告生成后仍然跳到底部（滚动修复不彻底）

**现象**：问题 2 的修复（`showReport` 滚到报告顶部）在实际测试中无效，报告出来后页面还是停在最底部。

**根因**：`done` 事件处理中，`showReport()` 滚到报告顶部后，紧接着调用了 `addMessage('✅ 研究完成...')`（默认 `scroll=true`），这条消息立刻又把页面拉回底部。两次滚动顺序不对，后者覆盖了前者。

**修复**：调整 `done` 事件中的执行顺序，并将完成消息标记为不滚动：

```javascript
case 'done':
  stopTimer();
  completeAllSteps();
  document.getElementById('research-btn').disabled = false;
  if (msg.data.report) {
    // 先渲染报告并滚到顶部
    showReport(msg.data.report);
    // 再加完成消息，但不滚动（scroll=false）
    addMessage('assistant', `✅ 研究完成！...`, false);
  }
```

关键改动：`addMessage` 加了 `false` 参数，确保完成提示不会覆盖报告的滚动位置。

**涉及文件**：`src/frontend/index.html`

---

### 问题 6：对话导航圆点 tooltip 抖动（鬼畜效果）

**现象**：右侧对话导航圆点，鼠标悬停时 tooltip 剧烈闪烁/抽搐，无法正常显示问题内容。

**根因**：三层问题叠加。

1. **CSS overflow 裁剪**：`.chat-nav` 设置了 `overflow-y: auto`，tooltip 作为 `.chat-nav-dot` 的子元素，超出导航容器后被裁剪不可见。

2. **scale 动画干扰**：圆点 hover 时有 `transform: scale(1.4)` 动画，元素尺寸变化导致 hover 区域不稳定，反复触发 enter/leave 事件。

3. **tooltip 与 hover 区域重叠**：tooltip 定位在圆点左侧（`right: 20px`），部分覆盖了圆点本身。鼠标从圆点移向 tooltip 时触发 `mouseleave` → tooltip 消失 → 鼠标又回到圆点 → `mouseenter` → tooltip 出现 → 循环。

**修复**：

1. **tooltip 改为 `position: fixed`** — 从 `<body>` 直接挂载，脱离所有 `overflow` 容器：

```html
<body>
<div id="nav-tooltip"></div>  <!-- 固定定位，不受父容器 overflow 影响 -->
```

2. **去掉圆点的 scale 动画** — 避免 hover 区域抖动：

```css
.chat-nav-dot:hover { background: var(--accent); border-color: var(--accent); }
/* 移除了 transform: scale(1.4) */
```

3. **tooltip 定位到圆点正上方** — 不与 hover 区域重叠，加 100ms 延迟隐藏防止快速切换：

```javascript
dot.addEventListener('mouseenter', () => {
  clearTimeout(tooltipTimer);
  const rect = dot.getBoundingClientRect();
  tooltip.style.top = (rect.top - 36) + 'px';   // 圆点上方 36px
  tooltip.style.left = (rect.left + rect.width/2) + 'px';
  tooltip.style.transform = 'translateX(-50%)';   // 水平居中
  tooltip.classList.add('visible');
});
dot.addEventListener('mouseleave', () => {
  tooltipTimer = setTimeout(() => tooltip.classList.remove('visible'), 100);
});
```

**涉及文件**：`src/frontend/index.html`

---

### 问题 7：前端 UI 设计迭代（三轮重构）

**现象**：初始 UI 为科幻/霓虹风格（深色背景、发光效果、粒子动画），用户反馈"丑"、"太简陋"。

**迭代过程**：

1. **第一版 — 科幻霓虹**：深色 `#060a14` 背景、gradient mesh、backdrop-filter blur、SVG 发光图标 → 用户评价"丑"
2. **第二版 — Glassmorphism**：半透明卡片、模糊背景、发光边框 → 用户评价"背景 UI 一样的丑"
3. **第三版 — ChatGPT 暗色极简**：`#212121` 纯深灰背景、无边框、无发光、圆形发送按钮、角色头像 → 用户评价"为什么一定要黑的灰的，浅色系一点"
4. **最终版 — 浅色系极简**：白底 `#f7f7f8`、深色文字 `#0f0f17`、浅灰侧边栏 `#efeff1`、无粒子/发光/渐变

**最终配色**：

```css
:root {
  --bg: #f7f7f8;           /* 主背景：浅灰 */
  --bg-sidebar: #efeff1;   /* 侧边栏：更浅的灰 */
  --text: #0f0f17;         /* 主文字：深色 */
  --text-secondary: #3c3c4a;
  --text-muted: #8b8b9e;
  --border: #d4d4de;       /* 边框：极淡 */
  --accent: #6d5bd0;       /* 强调色：紫色 */
  --success: #16a34a;
}
```

**设计原则**：参考 ChatGPT 界面，拒绝粒子/发光/渐变/科幻元素，保持干净留白，信息密度适中。

**涉及文件**：`src/frontend/index.html`（三次完整重写）

---

### 改动总览

| 文件 | 改动内容 |
|------|---------|
| `src/agent/state.py` | 新增 `conversation_history: str` 字段 |
| `src/agent/planner.py` | 新增 `local_docs` 工具 + 追问识别规则 + 会话历史注入 prompt |
| `src/agent/writer.py` | 新增追问写作规则（语言转换/深入/新问题）+ 会话历史注入 prompt |
| `src/agent/researcher.py` | 新增 `_search_local_docs()` + `local_docs` 工具处理 + 自动 RAG 兜底检索 |
| `src/backend/websocket_handler.py` | 接收 `conversation_history`、格式化、`_summarize_history()` LLM 摘要压缩、`get_config` 导入 |
| `src/frontend/index.html` | `conversationHistory` 数组追踪、请求携带历史、报告滚到顶部（`done` 事件顺序修复）、`loadSession` 重建历史、`newResearch` 清空历史、对话导航圆点 + 固定定位 tooltip（去 scale 防抖）、浅色系极简 UI（三轮重构：科幻→glassmorphism→暗色极简→浅色系） |
