from pathlib import Path

from test_api import FakeLLM, make_client

from world_engine.infra.llm import LLMError

SUGGESTIONS = {
    "suggestions": [
        {
            "content": "驿道封锁的消息传来，官兵挨户搜查「佩戴皇家玉佩的人」。",
            "targets": ["沈雁", "路人甲"],
            "importance": 7,
            "reason": "把私下疑心逼成公开危机。",
        },
        {
            "content": "一个自称认得玉佩的旧货郎进店，开口要三倍价钱买下它。",
            "targets": ["柳娘"],
            "importance": 5,
            "reason": "给柳娘一个护人还是装糊涂的选择。",
        },
        {
            "content": "雪夜里茶馆的灯突然灭了，黑暗里有人握住了阿蛮的手腕。",
            "targets": [],
            "importance": 9,
            "reason": "把主动权交给在场所有人。",
        },
    ]
}


class SuggestLLM(FakeLLM):
    def __init__(self):
        self.prompts: list[str] = []

    def chat_json(self, messages, **kwargs):
        text = "\n".join(m["content"] for m in messages)
        self.prompts.append(text)
        if "导演参谋" in text:
            return SUGGESTIONS
        return super().chat_json(messages, **kwargs)


class BrokenLLM(FakeLLM):
    def chat_json(self, messages, **kwargs):
        if "导演参谋" in "\n".join(m["content"] for m in messages):
            raise LLMError("suggest failed")
        return super().chat_json(messages, **kwargs)


def test_suggest_returns_three_and_filters_unknown_targets(tmp_path: Path):
    llm = SuggestLLM()
    client = make_client(tmp_path, llm=llm)
    client.post("/api/worlds", json={"world_id": "w1"})

    resp = client.post("/api/worlds/w1/director/suggest")
    assert resp.status_code == 200
    suggestions = resp.json()["suggestions"]
    assert len(suggestions) == 3
    # names that are not in the roster are dropped, order is kept
    assert suggestions[0]["targets"] == ["沈雁"]
    assert suggestions[1]["targets"] == ["柳娘"]
    assert suggestions[2]["targets"] == []
    assert suggestions[0]["importance"] == 7
    assert suggestions[0]["reason"]
    # the prompt hands the model the real roster and recent state
    assert "沈雁" in llm.prompts[-1]
    assert "柳娘" in llm.prompts[-1]
    assert "当前回合：0" in llm.prompts[-1]


def test_suggest_never_mutates_the_world(tmp_path: Path):
    client = make_client(tmp_path, llm=SuggestLLM())
    client.post("/api/worlds", json={"world_id": "w1"})

    assert client.post("/api/worlds/w1/director/suggest").status_code == 200
    assert client.get("/api/worlds/w1/history").json() == []
    assert client.get("/api/worlds/w1").json()["turn"] == 0


def test_suggest_unknown_world_is_404(tmp_path: Path):
    client = make_client(tmp_path, llm=SuggestLLM())
    resp = client.post("/api/worlds/nope/director/suggest")
    assert resp.status_code == 404


def test_suggest_llm_error_is_502(tmp_path: Path):
    client = make_client(tmp_path, llm=BrokenLLM())
    client.post("/api/worlds", json={"world_id": "w1"})
    resp = client.post("/api/worlds/w1/director/suggest")
    assert resp.status_code == 502
