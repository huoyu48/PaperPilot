"""WebSocket handler: real-time research progress streaming."""

from __future__ import annotations

import asyncio
import json
import queue
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from src.agent.graph import create_agent
from src.memory.sessions import SessionStore
from src.report.generator import generate_report_file
from src.utils.config import get_config
from src.utils.logging import logger

NODE_LABELS = {
    "planner": "正在规划研究方案...",
    "researcher": "正在检索多源资料...",
    "synthesizer": "正在交叉分析文献...",
    "writer": "正在生成研究报告...",
    "reviewer": "正在审核报告质量...",
}

# Cap conversation history to ~12K tokens (~48K chars)
# When exceeded, older exchanges are LLM-summarized; recent ones kept in full.
MAX_HISTORY_CHARS = 48_000
SUMMARY_PROMPT = (
    "请用中文简洁总结以下对话历史，保留关键信息（用户问过什么、得到了什么结论、有哪些重要发现）。"
    "输出格式：先列要点，不超过 500 字。\n\n"
)


async def _summarize_history(history_str: str) -> str:
    """If history is too long, LLM-summarize older exchanges and keep recent ones intact."""
    if len(history_str) <= MAX_HISTORY_CHARS:
        return history_str

    # Split: find a clean boundary roughly in the middle
    mid = len(history_str) // 2
    cut = history_str.find("\n\n[User]:", mid)
    if cut < 0:
        cut = mid
    old_part = history_str[:cut].strip()
    recent_part = history_str[cut:].strip()

    # Summarize the old part via LLM
    try:
        from src.llm.client import create_llm
        from langchain_core.messages import SystemMessage

        cfg = get_config()
        llm = create_llm(
            cfg.llm_provider, cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model,
        )
        resp = await asyncio.to_thread(
            llm.invoke,
            [SystemMessage(content=SUMMARY_PROMPT), ("human", old_part)],
        )
        summary = resp.content.strip()
        logger.info(f"History summarized: {len(old_part)} chars → {len(summary)} chars")
        return f"[历史对话摘要]\n{summary}\n\n---\n\n[最近对话]\n{recent_part}"
    except Exception as exc:
        logger.warning(f"History summarization failed: {exc}, falling back to truncation")
        return "[Earlier conversation truncated to save context]\n\n" + recent_part


async def research_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time research progress."""
    await websocket.accept()
    logger.info("WebSocket: client connected")

    try:
        msg = await websocket.receive_json()
        query = msg.get("query", "")
        session_id = msg.get("session_id", uuid.uuid4().hex[:12])
        is_followup = bool(msg.get("session_id"))
        # Conversation history from frontend: [{role, content, report?}]
        raw_history: list[dict] = msg.get("conversation_history", [])

        if not query:
            await websocket.send_json({"event": "error", "data": "Missing 'query' field"})
            return

        logger.info(f"WebSocket: {'followup' if is_followup else 'new'} research '{query[:60]}' (session={session_id})")
        await websocket.send_json({"event": "session_start", "data": {"session_id": session_id}})

        # Build conversation_history string for planner/writer
        conversation_history = ""
        if raw_history:
            parts: list[str] = []
            for turn in raw_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                report = turn.get("report", "")
                if role == "user":
                    parts.append(f"[User]: {content}")
                elif report:
                    parts.append(f"[Assistant report]:\n{report}")
                else:
                    parts.append(f"[Assistant]: {content}")
            conversation_history = "\n\n".join(parts)
        elif is_followup:
            # Fallback: load from SessionStore if frontend didn't send history
            store = SessionStore()
            existing = store.get_session(session_id)
            if existing:
                parts = [f"[User]: {existing['query']}"]
                if existing.get("report"):
                    parts.append(f"[Assistant report]:\n{existing['report']}")
                for f in existing.get("followups", []):
                    parts.append(f"[User]: {f['query']}")
                    if f.get("report"):
                        parts.append(f"[Assistant report]:\n{f['report']}")
                conversation_history = "\n\n".join(parts)

        agent = create_agent()
        initial_state = {
            "query": query,
            "session_id": session_id,
            "conversation_history": await _summarize_history(conversation_history),
            "sub_queries": [],
            "research_results": [],
            "synthesis": "",
            "report": "",
            "review_feedback": "",
            "needs_revision": False,
            "revision_count": 0,
            "messages": [],
            "errors": [],
        }

        # Use a thread-safe queue to pass events from stream thread to async loop
        event_queue: queue.Queue = queue.Queue()
        final_state = dict(initial_state)

        def run_stream():
            nonlocal final_state
            current_node = None
            for event in agent.stream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    # Send node_complete event
                    event_queue.put(("node_complete", node_name))

                    # If there's a plan, send it immediately
                    if "sub_queries" in node_output:
                        event_queue.put(("plan", node_output["sub_queries"]))
                    if "research_results" in node_output:
                        event_queue.put(("results_count", len(node_output["research_results"])))
                    if "report" in node_output and node_output["report"]:
                        event_queue.put(("report", node_output["report"]))

                    # Merge output into final state
                    for k, v in node_output.items():
                        if k == "messages":
                            final_state.setdefault("messages", []).extend(v)
                        else:
                            final_state[k] = v

            event_queue.put(("done", None))

        # Start stream in thread
        stream_thread = asyncio.get_event_loop().run_in_executor(None, run_stream)

        # Send initial status
        await websocket.send_json({
            "event": "node_start",
            "data": {"node": "planner", "message": NODE_LABELS["planner"]},
        })

        # Poll the queue and send events to client
        last_node = None
        while True:
            try:
                event_type, event_data = event_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.2)
                # Check if stream is done
                if stream_thread.done():
                    break
                continue

            if event_type == "node_complete":
                await websocket.send_json({
                    "event": "node_complete",
                    "data": {"node": event_data, "message": f"✓ {NODE_LABELS.get(event_data, event_data)}"},
                })
                last_node = event_data

                # Send next node start if applicable
                node_order = ["planner", "researcher", "synthesizer", "writer", "reviewer"]
                if event_data in node_order:
                    idx = node_order.index(event_data)
                    if idx + 1 < len(node_order):
                        next_node = node_order[idx + 1]
                        await websocket.send_json({
                            "event": "node_start",
                            "data": {"node": next_node, "message": NODE_LABELS[next_node]},
                        })

            elif event_type == "plan":
                await websocket.send_json({"event": "plan", "data": event_data})
            elif event_type == "results_count":
                await websocket.send_json({"event": "results", "data": {"count": event_data}})
            elif event_type == "report":
                await websocket.send_json({"event": "report", "data": event_data})
            elif event_type == "done":
                break

        # Final done event
        report_text = final_state.get("report", "")
        sub_queries = final_state.get("sub_queries", [])

        report_path = generate_report_file(
            report_text,
            final_state.get("research_results", []),
            session_id,
        )

        # Persist session
        store = SessionStore()
        if is_followup:
            store.save_followup(session_id, query, report_text)
        else:
            store.save_session(session_id, query, sub_queries, report_text)

        await websocket.send_json({
            "event": "done",
            "data": {
                "session_id": session_id,
                "report": report_text,
                "report_path": report_path,
            },
        })

    except WebSocketDisconnect:
        logger.info("WebSocket: client disconnected")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        try:
            await websocket.send_json({"event": "error", "data": str(exc)})
        except Exception:
            pass
