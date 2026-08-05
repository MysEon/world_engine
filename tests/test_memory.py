import pytest

from world_engine.core.models import MemoryRecord
from world_engine.memory.retrieval import relevance, retrieve


def test_relevance_cjk_shares_grams():
    assert relevance("茶馆老板娘", "茶馆老板娘的消息") > 0.2
    assert relevance("矿洞里的宝藏", "今日天气晴朗") == 0.0


def test_relevance_english_words():
    assert relevance("treasure hunt", "treasure map") > relevance("treasure hunt", "weather")


def test_retrieve_favors_important_fresh_relevant():
    current = 20
    memories = [
        MemoryRecord(character="a", content="我和柳娘谈起山贼的动向", importance=8, turn=19),
        MemoryRecord(character="a", content="今天吃了碗面，味道一般", importance=1, turn=19),
        MemoryRecord(character="a", content="很久以前听说宝藏的事", importance=9, turn=1),
        MemoryRecord(character="a", content="柳娘泡茶的手艺很细致", importance=3, turn=10),
    ]
    result = retrieve("柳娘 山贼", memories, current, top_k=3)
    assert result[0].content == memories[0].content


def test_retrieve_respects_top_k():
    memories = [
        MemoryRecord(character="a", content=f"关于山贼的第{i}条记忆", importance=7, turn=i)
        for i in range(1, 10)
    ]
    result = retrieve("山贼", memories, current_turn=10, top_k=3)
    assert len(result) == 3
