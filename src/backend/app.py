"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response

from src.backend.routes import router
from src.backend.websocket_handler import research_websocket
from src.utils.config import get_config
from src.utils.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    setup_logging(cfg.log_level)
    logger.info("PaperPilot starting...")

    for d in ["data/chroma", "data/memory", "data/uploads", "data/reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info(f"PaperPilot ready — http://{cfg.host}:{cfg.port}")
    yield
    logger.info("PaperPilot shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PaperPilot",
        description="AI Research Assistant — Multi-Agent research orchestration",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router, prefix="/api")
    app.websocket("/ws/research")(research_websocket)

    @app.get("/")
    async def serve_frontend():
        html_path = Path(__file__).parent.parent / "frontend" / "index.html"
        if html_path.exists():
            html_content = html_path.read_text(encoding="utf-8")
            return Response(content=html_content, media_type="text/html")
        return Response(content="<h1>Frontend not found</h1>", media_type="text/html")

    return app
