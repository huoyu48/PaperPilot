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

### 问题 8：研究流程耗时过长（规划 2 分钟+）

**现象**：用户提问后，规划阶段耗时超过 2 分钟，整体流程经常超过 5 分钟。对比 ChatGPT 几秒出结果，体验差距明显。

**根因分析**：

1. **Planner 不必要地调用 LLM**：对于简单新查询（如"介绍下 agent 最新发展"），完全可以用规则生成子查询，但每次都调用 DeepSeek API，而 API 响应慢时耗时 1-2 分钟。
2. **子查询串行执行**：4 个 arxiv/web_search 子查询逐个执行，每个 3-10 秒，总耗时 15-40 秒。实际上它们之间无依赖，可以并行。
3. **Synthesizer + Writer 两次 LLM 调用**：synthesizer 做交叉分析，writer 再拿分析结果+同样的源数据写报告，两次调用有大量重复输入。
4. **ArXiv 强制 1 秒/结果延迟**：`arxiv.Client(delay_seconds=1.0)`，5 个结果 = 5 秒等待。
5. **LLM 参数不合理**：planner 只输出 ~200 token JSON，但 `max_tokens=4096`、`temperature=0.7`，高温导致更多重试。
6. **Writer 非流式输出**：用户要等整个报告生成完毕才能看到内容。

**修复**（分两轮优化）：

**第一轮：并行化 + 节点合并 + 参数调优**

1. **researcher.py** — `ThreadPoolExecutor` 并行执行所有子查询：
```python
with ThreadPoolExecutor(max_workers=min(len(pending), 5)) as pool:
    future_to_sq = {pool.submit(_execute_single, sq): sq for sq in pending}
    for future in as_completed(future_to_sq):
        results = future.result()
        all_results.extend(results)
```

2. **graph.py** — 合并 synthesizer 到 writer，5 节点变 4 节点：
```
planner → researcher → writer → reviewer  (原来 5 节点)
```

3. **arxiv_tool.py** — `delay_seconds` 1.0→0.05，`max_results` 5→3

4. **planner.py** — `temperature` 0.7→0.2，`max_tokens` 4096→256

5. **reviewer.py** — `temperature` 0.7→0.1，`max_tokens` 4096→256，修订上限 2→1

6. **client.py** — LLM 客户端缓存，复用 HTTP 连接

7. **websocket_handler.py** — 轮询间隔 200ms→50ms

8. **config.py** — `search_max_results` 5→3

**第二轮：规则化快速 Planner + Writer 流式输出**

1. **planner.py** — 新增 `_fast_plan()` 函数，基于关键词规则生成子查询，跳过 LLM 调用：
```python
def _fast_plan(query: str) -> list[SubQuery]:
    # 检测"最新/发展/趋势"→ arxiv + web_search
    # 检测"代码/实现/github"→ github
    # 检测"上传/这篇"→ local_docs
    # 规则生成 3-4 个子查询，0 秒完成

def planner_node(state):
    if not history:  # 新查询 → 规则化，不调 LLM
        return {"sub_queries": _fast_plan(query)}
    # 追问 → LLM（精简 prompt，~60 词）
```

2. **writer.py** — 流式输出，用户实时看到报告生成：
```python
_stream_callback = None  # 模块级回调

def writer_node(state):
    if _stream_callback:
        full_text = ""
        for chunk in llm.stream(messages):
            full_text += chunk.content
            _stream_callback(chunk.content)  # 推送到前端
    else:
        response = llm.invoke(messages)
```

3. **websocket_handler.py** — 设置流式回调，转发 `report_chunk` 事件

4. **frontend** — `report_chunk` 事件处理：渐进式 Markdown 渲染（`requestAnimationFrame` 节流）

**效果**：
- 规划阶段：2 分钟 → 0 秒（新查询）/ 10-20 秒（追问）
- 检索阶段：15-40 秒 → 3-10 秒（并行）
- 报告生成：等待 30-60 秒 → 实时流式输出，首字延迟 < 2 秒
- 整体感知速度提升约 70-80%

**涉及文件**：`src/agent/planner.py`, `src/agent/writer.py`, `src/agent/researcher.py`, `src/agent/graph.py`, `src/agent/reviewer.py`, `src/backend/websocket_handler.py`, `src/llm/client.py`, `src/tools/arxiv_tool.py`, `src/utils/config.py`, `src/frontend/index.html`

---

### 改动总览

| 文件 | 改动内容 |
|------|---------|
| `src/agent/state.py` | 新增 `conversation_history: str` 字段 |
| `src/agent/planner.py` | 新增 `local_docs` 工具 + 追问识别规则 + 会话历史注入 + 规则化快速 planner（新查询跳过 LLM）+ 精简 LLM prompt |
| `src/agent/writer.py` | 合并 synthesizer 逻辑 + 流式输出（`llm.stream()` + chunk 回调）+ 追问写作规则 + 会话历史注入 |
| `src/agent/researcher.py` | 新增 `_search_local_docs()` + `local_docs` 工具处理 + 自动 RAG 兜底检索 + `ThreadPoolExecutor` 并行执行子查询 |
| `src/agent/graph.py` | 合并 synthesizer 到 writer，5 节点变 4 节点 |
| `src/agent/reviewer.py` | 降低 temperature 0.7→0.1、max_tokens 4096→256、修订上限 2→1 |
| `src/backend/websocket_handler.py` | 接收 `conversation_history`、格式化、`_summarize_history()` LLM 摘要压缩、设置 writer 流式回调、转发 `report_chunk` 事件、轮询间隔 200ms→50ms |
| `src/llm/client.py` | LLM 客户端缓存（复用 HTTP 连接） |
| `src/tools/arxiv_tool.py` | `delay_seconds` 1.0→0.05、`max_results` 5→3 |
| `src/utils/config.py` | `search_max_results` 5→3 |
| `src/frontend/index.html` | `conversationHistory` 数组追踪、请求携带历史、报告滚到顶部、`loadSession` 重建历史、`newResearch` 清空历史、对话导航圆点 + 固定定位 tooltip、浅色系极简 UI（三轮重构）、折叠思考块、`report_chunk` 渐进式 Markdown 渲染（`requestAnimationFrame` 节流）、4 步 pipeline（移除综合步骤） |

---

### 问题 9：检索工具大面积失败导致研究流程卡死（网络环境适配）

**现象**：用户反馈"搜索资料很慢"，日志显示检索阶段耗时 3 分钟以上，其中 web_search 子查询每次都卡满 30 秒超时才返回，arxiv 全部返回 301 错误，RAG 检索报 "Cannot send a request, as the client has been closed"。

**根因分析**（三个独立问题叠加）：

1. **DuckDuckGo 搜索引擎不可达**：`web_search_tool.py` 依赖的 DuckDuckGo 在用户网络环境下连接超时（20s 无响应）。Jina Reader 和 Google 同样不可达。百度对 httpx 返回安全验证页（CAPTCHA），Bing 对中文查询处理异常（把中文句子拆成单字匹配）。`duckduckgo_search` 库也未安装（`pip install -e .` 被取消），走了 httpx fallback 但 fallback 端点同样连不上。

2. **ArXiv API HTTP→HTTPS 重定向未跟随**：`arxiv_tool.py` 用的是 `http://export.arxiv.org/api/query`，ArXiv 现在强制 301 重定向到 `https://`，但 httpx 默认不跟随重定向，导致所有 arxiv 查询直接失败，返回 0 结果。

3. **ChromaDB 客户端线程安全问题**：`ResearchVectorStore` 每次实例化都创建新的 Chroma 客户端。在 `ThreadPoolExecutor` 并行检索场景下，父线程的客户端被垃圾回收关闭，子线程使用时报 "client has been closed"。此外本地 HuggingFace embedding 模型（`BAAI/bge-small-zh-v1.5`）首次加载需要约 90 秒。

**修复**：

1. **`src/tools/web_search_tool.py`** — 完全重写，从 DuckDuckGo 切换到**搜狗搜索（Sogou）**：
   - 搜狗对 httpx 友好，不触发反爬验证
   - 中英文查询都能正确理解（不像 Bing 拆字）
   - 响应时间从 20-30 秒超时降到 0.5-1.5 秒
   - 用正则解析 HTML，无额外依赖

2. **`src/tools/arxiv_tool.py`** — URL 改 `https://` + 加 `follow_redirects=True`：
   ```python
   ARXIV_API = "https://export.arxiv.org/api/query"  # was http://
   resp = httpx.get(ARXIV_API, ..., follow_redirects=True)
   ```

3. **`src/rag/vectorstore.py`** — 改为单例模式，全局共享一个 ChromaDB 客户端：
   ```python
   _store: Chroma | None = None
   _store_lock = threading.Lock()

   def _get_store(collection_name, persist_dir) -> Chroma:
       global _store
       if _store is None:
           with _store_lock:
               if _store is None:
                   _store = Chroma(...)
       return _store
   ```

4. **`src/agent/researcher.py`** — RAG 检索改用 daemon 线程 + 10 秒硬超时，超时直接跳过不阻塞：
   ```python
   t = threading.Thread(target=_do_retrieve, daemon=True)
   t.start()
   t.join(timeout=10)
   if t.is_alive():
       return []  # 超时跳过，不阻塞 pipeline
   ```
   同时整体超时从 30s 降到 20s。

5. **`src/backend/app.py`** — 启动时在后台 daemon 线程预热向量库，不阻塞服务启动：
   ```python
   def _warmup():
       from src.rag.vectorstore import _get_store
       _get_store("paperpilot_research", cfg.chroma_path)

   threading.Thread(target=_warmup, daemon=True).start()
   ```

**效果**：

| 工具 | 修复前 | 修复后 |
|------|--------|--------|
| Arxiv | 失败（301 重定向） | 0.3-1.8s，3 篇论文 |
| WebSearch | 20-30s 超时，0 结果 | 0.5-1.5s，3 条结果 |
| RAG | 90s 卡死或 "client closed" 错误 | 10s 超时保护；启动后后台预热 |

**涉及文件**：`src/tools/web_search_tool.py`, `src/tools/arxiv_tool.py`, `src/rag/vectorstore.py`, `src/agent/researcher.py`, `src/backend/app.py`

---

### 问题 10：报告中的"最新研究"截止到 2025 年而非 2026 年

**现象**：2026 年 6 月底使用系统研究某个主题，生成的报告中"最新进展"部分仍然停留在 2025 年，完全没有 2026 年的内容。

**根因**：`planner.py` 顶部在 **import 时**就计算了年份常量：

```python
# 旧代码 — 模块级常量，进程启动后锁死
_YEAR_NOW = datetime.now().year
_YEAR_PREV = _YEAR_NOW - 1
_YEARS = f"{_YEAR_PREV} {_YEAR_NOW}"
```

这意味着：
1. 进程一旦启动，`_YEARS` 的值就固定不变，即使跨年也不会更新
2. uvicorn `--reload` 模式下，只有 Python 源码文件改动才会重启 worker；只要模块没改，进程一直存活，变量不会刷新
3. 如果服务是 2025 年启动的，到 2026 年子查询依然带 `"2024 2025"`，检索到的都是旧资料

**修复**：把模块级常量改为**函数式调用**，每次生成子查询时都重新读取当前年份：

```python
# 新代码 — 函数式，每次调用都拿实时年份
def _years_str() -> str:
    now = datetime.now().year
    return f"{now - 1} {now}"

def _year_now() -> int:
    return datetime.now().year
```

同时在 `writer.py` 的 system prompt 中注入当前日期，让 LLM 明确知道"今天是 2026-06-30"，对"最新/近期"的判断基于真实当前日期，而不是数据源中提到的旧日期：

```python
SYSTEM_PROMPT = f"""\
...
## Current Date Context

Today is {datetime.now().strftime("%B %d, %Y")}. When sources discuss "recent" or \
"latest" developments, interpret them relative to this date, not the dates mentioned \
in the sources. Prefer information from {datetime.now().year} and {datetime.now().year - 1}.
...
"""
```

**效果**：

| 时间 | 修复前子查询 | 修复后子查询 |
|------|-------------|-------------|
| 2026-06 | `xxx 最新进展 2024 2025` | `xxx 最新进展 2025 2026` |

**涉及文件**：`src/agent/planner.py`, `src/agent/writer.py`
