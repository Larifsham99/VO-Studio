"""Long-form chunking that keeps emotion-tag state across chunk boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .emotion_tags import Segment, parse

SENTENCE_RE = re.compile(r"(?<=[.!?…;])\s+|\n{2,}")


@dataclass
class Chunk:
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text)

    @property
    def tags(self) -> list[str]:
        """Dominant tags of the chunk (first segment's state + all events)."""
        first = self.segments[0]
        tags = [t for t in (first.emotion, first.volume) if t]
        for s in self.segments:
            tags += [e for e in s.events if e not in tags]
        return tags

    @property
    def uniform(self) -> bool:
        keys = {(s.emotion, s.volume) for s in self.segments}
        return len(keys) == 1


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in SENTENCE_RE.split(text) if p and p.strip()]


def chunk_text(text: str, max_chars: int = 300) -> list[Chunk]:
    """Pack sentences into chunks <= max_chars.

    A change of emotion/volume state always starts a new chunk, so each model
    call receives a single consistent delivery instruction. Sentences longer
    than max_chars are split at commas, then hard-wrapped at spaces.
    """
    pieces: list[Segment] = []
    for seg in parse(text).segments:
        sentences = split_sentences(seg.text) or [""]
        first = True
        for sent in sentences:
            for part in _split_long(sent, max_chars):
                # events belong only to the first piece of their segment
                pieces.append(Segment(part, seg.emotion, seg.volume, seg.events if first else []))
                first = False

    chunks: list[Chunk] = []
    cur: list[Segment] = []
    cur_len = 0
    for p in pieces:
        state_change = cur and (p.emotion, p.volume) != (cur[-1].emotion, cur[-1].volume)
        too_long = cur and cur_len + len(p.text) + 1 > max_chars
        if state_change or too_long or (p.events and cur):
            chunks.append(Chunk(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p.text) + 1
    if cur:
        chunks.append(Chunk(cur))
    return chunks


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    out: list[str] = []
    buf = ""
    for clause in re.split(r"(?<=,)\s+", sentence):
        if len(clause) > max_chars:
            words = clause.split()
            for w in words:
                if buf and len(buf) + len(w) + 1 > max_chars:
                    out.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            continue
        if buf and len(buf) + len(clause) + 1 > max_chars:
            out.append(buf)
            buf = clause
        else:
            buf = f"{buf} {clause}".strip()
    if buf:
        out.append(buf)
    return out
