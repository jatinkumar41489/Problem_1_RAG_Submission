"""FastAPI application factory.  Run:  uvicorn app.main:app --port 8000"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings
from app.exceptions import ConfigError, GenerationError, RAGError
from app.logging_config import get_logger, log_event, setup_logging
from app.services import Services

log = get_logger("api")


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or (services.settings if services else Settings.from_env())  # validates on startup
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "services"):
            app.state.services = services or Services(settings)
        log_event(log, "startup", config=settings.public_dict())
        yield
        app.state.services.close()

    app = FastAPI(title="Cost-Efficient RAG Service", version="1.0.0", lifespan=lifespan,
                  description="Grounded, cited QA over PDF/HTML/Markdown on ChromaDB.")
    if services is not None:
        app.state.services = services

    @app.exception_handler(RAGError)
    async def _rag_error(request: Request, exc: RAGError):
        status = 503 if isinstance(exc, GenerationError) else exc.http_status
        return JSONResponse(status_code=status, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errs = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": "Invalid request", "details": errs}})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = uuid.uuid4().hex[:12]
        log.error("unhandled_exception", extra={"event": "unhandled_exception", "error_id": rid,
                                                "exc_type": type(exc).__name__}, exc_info=True)
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error",
                            "message": f"Internal server error (error_id={rid})."}})

    app.include_router(router)
    return app


def _build_default_app() -> FastAPI:
    try:
        return create_app()
    except ConfigError as exc:  # clear startup failure
        raise SystemExit(f"Configuration error: {exc.message}")


app = _build_default_app()
