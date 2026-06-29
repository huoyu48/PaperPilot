"""PaperPilot entry point."""

import uvicorn

from src.backend.app import create_app
from src.utils.config import get_config

app = create_app()

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "main:app",
        host=cfg.host,
        port=cfg.port,
        reload=True,
    )
