from __future__ import annotations

import math
import re
from collections import Counter

from ..core.models import MemoryRecord

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[^\s\W]")
_ALPHA_RE = re.compile(r"[a-zA-Z]")


def _ngrams(text: str, n: int = 2) -> list[str]:
    """Character n-grams so relevance works for CJK text (no word boundaries)."""
    tokens = _TOKEN_RE.findall(text.lower())
    if _ALPHA_RE.search(text):
        grams: list[str] = []
        for tok in tokens:
            grams.extend(tok[i : i + n] for i in range(max(1, len(tok) - n + 1)))
        return grams
    joined = "".join(tokens)
    return [joined[i : i + n] for i in range(max(1, len(joined) - n + 1))]


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def relevance(query: str, content: str) -> float:
    """Character bigram Jaccard similarity between query and memory content."""
    return _jaccard(_ngrams(query), _ngrams(content))


def is_duplicate(content: str, memories: list[MemoryRecord], threshold: float = 0.55) -> bool:
    """True if a similar memory was already recorded (write-time consolidation)."""
    for m in memories:
        if m.archived:
            continue
        if relevance(content, m.content) >= threshold:
            return True
    return False


def recency_score(turns_ago: int, decay: float = 0.12) -> float:
    """Exponential time decay: fresher memories score higher."""
    return math.exp(-decay * turns_ago)


def importance_score(importance: int) -> float:
    return importance / 10.0


def retrieve(
    query: str,
    memories: list[MemoryRecord],
    current_turn: int,
    top_k: int = 6,
) -> list[MemoryRecord]:
    """Three-factor retrieval: recency x importance x relevance (Park et al., 2023)."""
    scored: list[tuple[float, MemoryRecord]] = []
    for m in memories:
        r = relevance(query, m.content)
        if r <= 0.0:
            continue
        score = (
            recency_score(max(0, current_turn - m.turn))
            * importance_score(m.importance)
            * r
        )
        scored.append((score, m))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [m for _, m in scored[:top_k]]
