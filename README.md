# PaperPilot — AI 科研助手

> 基于 LangGraph 多 Agent 协作的智能研究助手，通过学术论文、代码仓库和网络资源的多源检索，自动生成带引用的研究报告。

## 项目概述

PaperPilot 是一个面向科研场景的 AI Agent 系统。用户输入研究问题后，系统自动将问题拆解为多个子查询，分发到 Arxiv、GitHub、DuckDuckGo 等工具并行检索，交叉分析后生成结构化的 Markdown 研究报告。支持上传本地文档（PDF/TXT/MD/DOCX）进行 RAG 检索，支持多轮追问并保持完整会话上下文。

**技术栈：** LangGraph · FastAPI · WebSocket · ChromaDB · MCP Protocol · DeepSeek/OpenAI-compatible LLM

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (单文件 HTML)                 │
│         Dark Theme · 中文 UI · WebSocket 实时进度          │
└──────────────────────────┬──────────────────────────────┘
                           │ WebSocket / REST API
┌──────────────────────────▼──────────────────────────────┐
│                   FastAPI Backend                        │
│   /api/research  /api/sessions  /api/documents  /ws     │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                LangGraph StateGraph                      │
│                                                          │
│  ┌──────────┐  ┌────────────┐  ┌─────────────┐          │
│  │ Planner  │→ │ Researcher │→ │ Synthesizer │          │
│  │ 问题拆解  │  │ 多源检索    │  │ 交叉分析     │          │
│  └──────────┘  └────────────┘  └──────┬──────┘          │
│                                        │                 │
│  ┌──────────┐  ┌────────────┐         │                 │
│  │ Reviewer │← │   Writer   │←────────┘                 │
│  │ 质量审核  │  │ 报告生成    │                           │
│  └────┬─────┘  └────────────┘                           │
│       │ revise (max 2)                                   │
│       └──────→ Writer                                    │
└─────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                     Tool Layer                           │
│  Arxiv · GitHub · DuckDuckGo · Jina Reader · Local RAG  │
│                      MCP Server                          │
└─────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   Storage Layer                          │
│    ChromaDB (向量) · SQLite (会话/记忆) · File System     │
└─────────────────────────────────────────────────────────┘
```

## 核心特性

### 5 节点 LangGraph 工作流

| 节点 | 职责 | 输入 → 输出 |
|------|------|-------------|
| **Planner** | 将研究问题拆解为 3-5 个子查询 | query → `list[SubQuery]` |
| **Researcher** | 执行子查询，调用工具检索 | sub_queries → `list[ResearchResult]` |
| **Synthesizer** | 跨源交叉分析，识别主题/矛盾/缺口 | results → synthesis text |
| **Writer** | 生成结构化 Markdown 报告 | synthesis → report (1500-3000 words) |
| **Reviewer** | 5 维度评分，不达标触发修订 | report → approve / revise (max 2) |

### 多源检索工具

| 工具 | 数据源 | 用途 |
|------|--------|------|
| `arxiv` | Arxiv API | 学术论文（理论、方法、综述） |
| `github` | GitHub API | 代码仓库（实现、基准测试） |
| `web_search` | DuckDuckGo | 通用搜索（新闻、行业趋势） |
| `jina_reader` | Jina Reader API | 指定 URL 内容提取 |
| `local_docs` | ChromaDB | 用户上传文档的 RAG 检索 |

### RAG 管道

- **文档解析**：PDF（pdfplumber / pypdf）、TXT、Markdown、DOCX
- **文本分块**：RecursiveCharacterTextSplitter，中文友好分隔符（含"。"），chunk_size=512，overlap=50
- **向量存储**：ChromaDB，本地 embedding（BAAI/bge-small-zh-v1.5）或远程 API
- **会话隔离**：支持按 session_id 过滤检索范围
- **自动兜底**：每次研究自动检索向量库，即使 planner 未分配 local_docs 子查询

### 会话记忆系统

- **前端追踪**：`conversationHistory` 数组记录所有用户提问和完整报告
- **后端格式化**：转为 `[User]` / `[Assistant report]` 字符串注入 planner 和 writer 的 prompt
- **LLM 摘要压缩**：历史超过 48K 字符（≈12K tokens）时，用 LLM 将旧对话压缩为中文摘要（≤500 字），最近轮次保留完整原文
- **SQLite 持久化**：sessions 表 + followups 表，支持历史浏览和追问

### 实时 WebSocket 流式反馈

- 基于 `queue.Queue` 的同步 LangGraph stream → 异步 WebSocket 桥接
- 8 种事件类型：`session_start`、`node_start`、`node_complete`、`plan`、`results`、`report`、`done`、`error`
- 中文进度标签 + 实时计时器
- 对话导航圆点（hover 显示问题摘要，点击跳转，滚动高亮）

### MCP Server

通过 Model Context Protocol 暴露 4 个检索工具，支持外部 MCP 客户端调用：

```bash
python -m src.tools.mcp_server.server
```

## 快速开始

### 环境要求

- Python >= 3.11
- DeepSeek API Key（或其他 OpenAI 兼容 LLM）

### 安装

```bash
# 克隆仓库
git clone https://github.com/huoyu48/aperPilot.git
cd aperPilot

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e .
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入必要的环境变量：

```env
# 必填：LLM API Key
PAPERPILOT_LLM_API_KEY=sk-your-key-here

# 可选：切换 LLM 提供商（兼容所有 OpenAI 格式 API）
PAPERPILOT_LLM_BASE_URL=https://api.deepseek.com/v1
PAPERPILOT_LLM_MODEL=deepseek-chat

# 可选：GitHub Token（提升 API 速率限制）
PAPERPILOT_GITHUB_TOKEN=ghp-your-token

# 可选：服务端口（默认 8000）
PAPERPILOT_PORT=8002
```

### 启动

```bash
python main.py
```

浏览器打开 `http://localhost:8002`（或你配置的端口）。

## 项目结构

```
paperpilot/
├── main.py                     # 入口：uvicorn 启动
├── pyproject.toml              # 依赖管理
├── .env.example                # 环境变量模板
├── docs/
│   └── paperpilot_改进记录.md   # 开发问题修复记录
└── src/
    ├── agent/                  # LangGraph 工作流
    │   ├── graph.py            # 状态图构建（5 节点 + 条件边）
    │   ├── state.py            # AgentState / SubQuery / ResearchResult
    │   ├── planner.py          # 问题拆解 + 追问识别
    │   ├── researcher.py       # 工具调度 + RAG 兜底检索
    │   ├── synthesizer.py      # 跨源交叉分析
    │   ├── writer.py           # 报告生成 + 语言切换
    │   └── reviewer.py         # 质量评分 + 修订决策
    ├── backend/                # FastAPI 后端
    │   ├── app.py              # 应用初始化 + 生命周期
    │   ├── routes.py           # REST API 端点
    │   ├── websocket_handler.py # WebSocket 实时流 + 历史压缩
    │   └── schemas.py          # Pydantic 请求/响应模型
    ├── frontend/
    │   └── index.html          # 单文件前端（暗色主题中文 UI）
    ├── llm/
    │   └── client.py           # LLM / Embedding 工厂函数
    ├── memory/                 # 三层记忆架构
    │   ├── short_term.py       # 滑动窗口短期记忆
    │   ├── sessions.py         # SQLite 会话持久化
    │   ├── long_term.py        # 研究画像长期记忆
    │   └── summary.py          # LLM 摘要记忆
    ├── rag/                    # RAG 检索管道
    │   ├── parser.py           # 文档解析（PDF/TXT/MD/DOCX）
    │   ├── chunker.py          # 文本分块
    │   ├── embeddings.py       # Embedding 封装
    │   ├── vectorstore.py      # ChromaDB 向量存储
    │   └── retriever.py        # 检索接口
    ├── report/
    │   ├── generator.py        # Markdown 报告文件生成
    │   └── citations.py        # 引用管理
    ├── tools/                  # 检索工具
    │   ├── arxiv_tool.py       # Arxiv 论文搜索
    │   ├── github_tool.py      # GitHub 仓库搜索
    │   ├── web_search_tool.py  # DuckDuckGo 网页搜索
    │   ├── jina_reader.py      # URL 内容提取
    │   └── mcp_server/         # MCP 协议服务端
    └── utils/
        ├── config.py           # pydantic-settings 配置
        └── logging.py          # loguru 日志
```

## API 接口

### REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/research` | 启动研究（支持 session_id 追问） |
| `GET` | `/api/sessions` | 列出所有历史会话 |
| `GET` | `/api/sessions/{id}` | 获取会话详情（含追问记录） |
| `DELETE` | `/api/sessions/{id}` | 删除会话 |
| `POST` | `/api/documents` | 上传文档到 RAG 知识库 |
| `GET` | `/api/memory` | 获取研究画像 |

### WebSocket

连接 `ws://localhost:8002/ws/research`，发送：

```json
{
  "query": "知识追踪最新研究动态",
  "session_id": "可选-追问时传入",
  "conversation_history": [
    {"role": "user", "content": "之前的问题"},
    {"role": "assistant", "report": "之前的完整报告"}
  ]
}
```

服务端事件流：

```
session_start → node_start(planner) → node_complete → plan
→ node_start(researcher) → results → node_complete
→ node_start(synthesizer) → node_complete
→ node_start(writer) → node_complete
→ node_start(reviewer) → node_complete → done
```

## 配置参考

所有环境变量以 `PAPERPILOT_` 为前缀：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | `""` | LLM API 密钥（必填） |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | API 地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `TEMPERATURE` | `0.7` | 生成温度 |
| `MAX_TOKENS` | `4096` | 单次输出上限 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | Embedding 模型 |
| `GITHUB_TOKEN` | `""` | GitHub PAT |
| `CHROMA_PATH` | `data/chroma` | 向量库路径 |
| `CHUNK_SIZE` | `512` | 分块大小 |
| `CHUNK_OVERLAP` | `50` | 分块重叠 |
| `MAX_MEMORY_TURNS` | `10` | 短期记忆轮次 |
| `PORT` | `8000` | 服务端口 |

## License

MIT
