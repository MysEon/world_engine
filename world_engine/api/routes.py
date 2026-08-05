from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from .manager import WorldManager
from .schemas import (
    AgentView,
    CreateWorldRequest,
    NovelView,
    StepRequest,
    StepResponse,
    WorldSummary,
    WriteRequest,
    to_novel_view,
    to_world_summary,
)

router = APIRouter()


def _manager(request: Request) -> WorldManager:
    return request.app.state.manager


@router.get("/worlds", response_model=list[str])
def list_worlds(request: Request):
    return _manager(request).world_ids()


@router.post("/worlds", response_model=WorldSummary, status_code=201)
def create_world(request: Request, body: CreateWorldRequest):
    engine = _manager(request).get(body.world_id, body.preset_path)
    return to_world_summary(body.world_id, engine.world)


@router.get("/worlds/{world_id}", response_model=WorldSummary)
def world_status(world_id: str, request: Request):
    engine = _manager(request).get(world_id)
    return to_world_summary(world_id, engine.world)


@router.post("/worlds/{world_id}/step", response_model=StepResponse)
def step(world_id: str, request: Request, body: StepRequest | None = None):
    manager = _manager(request)
    engine = manager.get(world_id)
    body = body or StepRequest()
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
    engine = _manager(request).get(world_id)
    events = engine.store.events(world_id, limit=limit)
    return [
        {"turn": e.turn, "kind": e.kind, "actor": e.actor, "content": e.content, "created_at": e.created_at}
        for e in events
    ]


@router.post("/worlds/{world_id}/novel/plan", response_model=NovelView)
def plan_novel(world_id: str, request: Request):
    engine = _manager(request).get(world_id)
    novel = engine.plan_novel()
    return to_novel_view(novel)


@router.post("/worlds/{world_id}/novel/write", response_model=NovelView)
def write_novel(world_id: str, request: Request, body: WriteRequest | None = None):
    engine = _manager(request).get(world_id)
    body = body or WriteRequest()
    novel = engine.write_novel(body.chapters)
    return to_novel_view(novel)


@router.get("/worlds/{world_id}/novel", response_model=NovelView)
def get_novel(world_id: str, request: Request):
    manager = _manager(request)
    novel = manager.get(world_id).store.load_novel(world_id)
    if novel is None:
        raise HTTPException(status_code=404, detail="该世界还没有小说，先调用 /novel/plan")
    return to_novel_view(novel)


@router.get("/worlds/{world_id}/novel/export")
def export_novel(world_id: str, request: Request):
    manager = _manager(request)
    markdown = manager.get(world_id).export_novel()
    return Response(content=markdown, media_type="text/markdown; charset=utf-8")
