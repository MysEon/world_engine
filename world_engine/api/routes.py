from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from ..infra.config import validate_provider
from ..infra.llm import _sanitize, test_llm_connection
from ..infra.llm_settings import RuntimeLLMConfig, build_effective
from ..infra.logging import get_logger
from ..narrative.engine import WorldEngine
from .manager import WorldManager
from .schemas import (
    AgentView,
    CreateWorldRequest,
    LLMSettingsRequest,
    LLMSettingsResponse,
    LLMTestRequest,
    LLMTestResponse,
    NovelPlanRequest,
    NovelView,
    StepRequest,
    StepResponse,
    WorldSummary,
    WriteRequest,
    to_novel_view,
    to_world_summary,
)

logger = get_logger("api")

router = APIRouter()


def _manager(request: Request) -> WorldManager:
    return request.app.state.manager


def _existing_engine(request: Request, world_id: str) -> WorldEngine:
    """Resolve a persisted world or fail with 404 (GET must not create worlds)."""
    try:
        return _manager(request).get_existing(world_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"世界不存在：{world_id}") from None


@router.get("/worlds", response_model=list[str])
def list_worlds(request: Request):
    return _manager(request).world_ids()


@router.post("/worlds", response_model=WorldSummary, status_code=201)
def create_world(request: Request, body: CreateWorldRequest):
    engine = _manager(request).get(body.world_id, body.preset_path)
    return to_world_summary(body.world_id, engine.world)


@router.get("/worlds/{world_id}", response_model=WorldSummary)
def world_status(world_id: str, request: Request):
    engine = _existing_engine(request, world_id)
    return to_world_summary(world_id, engine.world)


@router.post("/worlds/{world_id}/step", response_model=StepResponse)
def step(world_id: str, request: Request, body: StepRequest | None = None):
    manager = _manager(request)
    body = body or StepRequest()
    with manager.lock(world_id):
        engine = _existing_engine(request, world_id)
        result = engine.step(director_event=body.event.to_core() if body.event else None)

        agents = []
        for name, out in result.outputs.items():
            new_memories = [
                m.content
                for m in engine.store.memories(world_id, name)
                if m.turn == result.turn
            ]
            agents.append(
                AgentView(name=name, thought=out.thought, speech=out.speech, action=out.action, memories=new_memories)
            )
    return StepResponse(turn=result.turn, narrative=result.narrative, agents=agents, events=result.event_log)


@router.get("/worlds/{world_id}/history")
def history(world_id: str, request: Request, limit: int = 20):
    engine = _existing_engine(request, world_id)
    events = engine.store.events(world_id, limit=limit)
    return [
        {"turn": e.turn, "kind": e.kind, "actor": e.actor, "content": e.content, "created_at": e.created_at}
        for e in events
    ]


@router.post("/worlds/{world_id}/novel/plan", response_model=NovelView)
def plan_novel(world_id: str, request: Request, body: NovelPlanRequest | None = None):
    engine = _existing_engine(request, world_id)
    try:
        novel = engine.plan_novel(force=body.force if body else False)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return to_novel_view(novel)


@router.post("/worlds/{world_id}/novel/write", response_model=NovelView)
def write_novel(world_id: str, request: Request, body: WriteRequest | None = None):
    engine = _existing_engine(request, world_id)
    body = body or WriteRequest()
    novel = engine.write_novel(body.chapters)
    return to_novel_view(novel)


@router.get("/worlds/{world_id}/novel", response_model=NovelView)
def get_novel(world_id: str, request: Request):
    engine = _existing_engine(request, world_id)
    novel = engine.store.load_novel(world_id)
    if novel is None:
        raise HTTPException(status_code=404, detail="该世界还没有小说，先调用 /novel/plan")
    return to_novel_view(novel)


@router.get("/worlds/{world_id}/novel/export")
def export_novel(world_id: str, request: Request):
    engine = _existing_engine(request, world_id)
    try:
        markdown = engine.export_novel()
    except KeyError:
        raise HTTPException(status_code=404, detail="该世界还没有小说，先调用 /novel/plan") from None
    return Response(content=markdown, media_type="text/markdown; charset=utf-8")


# -- LLM settings -------------------------------------------------------------


@router.get("/settings/llm", response_model=LLMSettingsResponse)
def get_llm_settings(request: Request):
    store = request.app.state.llm_store
    return LLMSettingsResponse(**store.public())


@router.put("/settings/llm", response_model=LLMSettingsResponse)
def put_llm_settings(request: Request, body: LLMSettingsRequest):
    if body.provider is not None:
        try:
            validate_provider(body.provider)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    store = request.app.state.llm_store
    patch = RuntimeLLMConfig(
        provider=body.provider,
        base_url=body.base_url,
        model=body.model,
        api_key=body.api_key,
    )
    public = store.update(patch)
    logger.info(
        "llm_settings_updated",
        provider=public["provider"],
        base_url=public["base_url"],
        model=public["model"],
        key_changed=body.api_key is not None,
    )
    return LLMSettingsResponse(**public)


@router.post("/settings/llm/test", response_model=LLMTestResponse)
def test_llm_settings(request: Request, body: LLMTestRequest):
    if body.provider is not None:
        try:
            validate_provider(body.provider)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    defaults = request.app.state.settings.llm
    cfg = RuntimeLLMConfig(
        provider=body.provider,
        base_url=body.base_url,
        model=body.model,
        api_key=body.api_key,
    )
    eff = build_effective(cfg, defaults)
    try:
        model, latency_ms = test_llm_connection(eff)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=_sanitize(str(exc), eff.api_key)) from exc
    return LLMTestResponse(ok=True, latency_ms=latency_ms, model=model)
