from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..infra.config import Settings, load_settings
from ..infra.llm import LLMClient, LLMError
from ..infra.logging import configure_logging
from .manager import WorldManager
from .middleware import RequestLogMiddleware
from .routes import router

#: Vite dev (5173) and preview (4173) origins allowed to call the API from a browser.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.logging.level, settings.logging.json_output)
    app = FastAPI(title="World Engine API", version="0.1.0", description="让 AI 角色自由生长，再把他们的故事写成小说。")
    app.state.settings = settings
    manager = WorldManager(settings, llm=llm)
    app.state.manager = manager
    app.state.llm_store = manager.llm_store

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(RequestLogMiddleware)
    app.include_router(router, prefix="/api")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        access_log=False,
    )
