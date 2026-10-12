"""Stage 5 (starter): a tiny knowledge base with BM25 keyword search.

Each markdown file in data/kb is split into chunks at "## " headings.
BM25 is old-school keyword ranking: no embeddings, no extra installs, and the
scores are easy to inspect. Upgrading to embeddings / hybrid search is a
month-2 exercise.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set(
    "a an the is are was were be been to of and or in on at for with my me your i you it its "
    "this that do does did how what when can could would should will there their they we our "
    "us if so as by from about any have has".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@dataclass
class Chunk:
    id: int
    source: str
    title: str
    text: str


class KnowledgeBase:
    def __init__(self, folder: str | Path, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        for path in sorted(Path(folder).glob("*.md")):
            self._load(path)
        self._tfs = [Counter(tokenize(c.title + " " + c.text)) for c in self.chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg = sum(self._lens) / max(1, len(self._lens))
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _load(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        doc_title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
        for part in re.split(r"^## ", text, flags=re.M)[1:]:
            heading, _, body = part.partition("\n")
            body = body.strip()
            if body:
                self.chunks.append(Chunk(len(self.chunks), path.name, f"{doc_title} > {heading.strip()}", body))

    def search(self, query: str, k: int = 3) -> list[tuple[float, Chunk]]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, dl in zip(self.chunks, self._tfs, self._lens):
            score = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    score += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avg))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda p: -p[0])
        return scored[:k]
