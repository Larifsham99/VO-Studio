"""Emotion tag parsing and per-model control mapping.

Tags never reach the model as literal text. They are parsed out and turned
into whatever control signal a given model actually understands:

* ``instruction`` - a natural-language style instruction (VoxCPM2
  "controllable cloning", MOSS-TTS, Qwen3-TTS instruct ...)
* ``exaggeration`` - Chatterbox's emotion-intensity scalar
* ``emotion bank`` - a reference clip of the *same permanent voice* performing
  that emotion (works with any zero-shot cloning model, because those models
  copy the delivery of their reference)

State tags ([sad], [whispers] ...) stay active until replaced by another tag of
the same group. Event tags ([laughs], [sighs]) apply only to the segment that
directly follows them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmotionSpec:
    name: str
    group: str  # "emotion" | "volume" | "event"
    instruction: str  # English style instruction for instruct-capable models
    arousal: float  # 0 (low energy) .. 1 (high energy)
    valence: float  # -1 negative .. +1 positive
    exaggeration: float  # Chatterbox emotion_adv, 0.25..1.0 useful range
    # Labels of emotion2vec_plus that count as a "hit" for automatic scoring.
    # Empty = no automatic target; judged by prosody checks / listening test.
    ser_targets: frozenset = field(default_factory=frozenset)


def _s(*labels: str) -> frozenset:
    return frozenset(labels)


EMOTIONS: dict[str, EmotionSpec] = {
    s.name: s
    for s in [
        EmotionSpec("excited", "emotion", "excited, energetic, fast and lively", 0.9, 0.8, 0.8, _s("happy", "surprised")),
        EmotionSpec("happy", "emotion", "happy, warm, smiling while speaking", 0.65, 0.9, 0.6, _s("happy")),
        EmotionSpec("sad", "emotion", "sad, low and heavy, slower pace", 0.25, -0.8, 0.5, _s("sad")),
        EmotionSpec("angry", "emotion", "angry, tense and forceful", 0.9, -0.8, 0.85, _s("angry")),
        EmotionSpec("curious", "emotion", "curious, inquisitive, rising intonation", 0.55, 0.3, 0.5),
        EmotionSpec("sarcastic", "emotion", "sarcastic, dry and mocking tone", 0.5, -0.3, 0.55),
        EmotionSpec("crying", "emotion", "crying, tearful, voice breaking", 0.5, -0.9, 0.8, _s("sad")),
        EmotionSpec("fearful", "emotion", "fearful, nervous and trembling", 0.7, -0.7, 0.7, _s("fearful")),
        EmotionSpec("calm", "emotion", "calm, relaxed and steady", 0.2, 0.3, 0.3, _s("neutral")),
        EmotionSpec("confident", "emotion", "confident, assertive and clear", 0.6, 0.5, 0.5),
        EmotionSpec("serious", "emotion", "serious, grave and measured", 0.4, -0.2, 0.4, _s("neutral")),
        EmotionSpec("dramatic", "emotion", "dramatic, theatrical with strong emphasis and pauses", 0.8, 0.0, 0.9),
        EmotionSpec("whispers", "volume", "whispering, breathy and very quiet", 0.1, 0.0, 0.3),
        EmotionSpec("softly", "volume", "speaking softly and gently", 0.2, 0.2, 0.3),
        EmotionSpec("shouts", "volume", "shouting loudly", 1.0, 0.0, 1.0),
        EmotionSpec("laughs", "event", "laughing", 0.8, 0.8, 0.8),
        EmotionSpec("sighs", "event", "sighing", 0.2, -0.3, 0.4),
    ]
}

ALIASES = {
    "whisper": "whispers", "whispering": "whispers",
    "shout": "shouts", "shouting": "shouts",
    "laugh": "laughs", "laughing": "laughs",
    "sigh": "sighs", "sighing": "sighs",
    "soft": "softly", "cry": "crying", "fear": "fearful", "scared": "fearful",
}

TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z _-]{0,30})\]")


def canonical(tag: str) -> str | None:
    t = tag.strip().lower().replace(" ", "_")
    t = ALIASES.get(t, t)
    return t if t in EMOTIONS else None


@dataclass
class Segment:
    text: str
    emotion: str | None = None  # active state tag of group "emotion"
    volume: str | None = None  # active state tag of group "volume"
    events: list[str] = field(default_factory=list)  # one-shot tags before this text

    @property
    def tags(self) -> list[str]:
        out = [t for t in (self.emotion, self.volume) if t]
        return out + list(self.events)


@dataclass
class ParseResult:
    segments: list[Segment]
    unknown_tags: list[str]

    @property
    def plain_text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text).strip()


def parse(text: str) -> ParseResult:
    """Split tagged text into segments with their active tags.

    Unknown bracketed words are removed from the spoken text and reported, so a
    typo like ``[exited]`` is never read aloud.
    """
    segments: list[Segment] = []
    unknown: list[str] = []
    state = {"emotion": None, "volume": None}
    pending_events: list[str] = []
    pos = 0

    def flush(chunk: str) -> None:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if not chunk:
            return
        segments.append(Segment(chunk, state["emotion"], state["volume"], list(pending_events)))
        pending_events.clear()

    for m in TAG_RE.finditer(text):
        flush(text[pos:m.start()])
        pos = m.end()
        name = canonical(m.group(1))
        if name is None:
            unknown.append(m.group(1))
            continue
        spec = EMOTIONS[name]
        if spec.group == "event":
            pending_events.append(name)
        else:
            state[spec.group] = name
    flush(text[pos:])
    if pending_events:  # trailing event tag, e.g. "... selamat tinggal [laughs]"
        if segments:
            segments.append(Segment("", state["emotion"], state["volume"], list(pending_events)))
        else:
            segments.append(Segment("", None, None, list(pending_events)))
    return ParseResult(segments, unknown)


def instruction_for(tags: list[str]) -> str:
    """Natural-language instruction for instruct-capable models."""
    parts = [EMOTIONS[t].instruction for t in tags if t in EMOTIONS]
    return ", ".join(parts)


def exaggeration_for(tags: list[str], default: float = 0.5) -> float:
    vals = [EMOTIONS[t].exaggeration for t in tags if t in EMOTIONS]
    return max(vals) if vals else default


def has_tags(text: str) -> bool:
    return bool(TAG_RE.search(text))
