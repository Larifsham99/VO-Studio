"""Malay text normalisation: numbers, currency, percentages, dates, ordinals,
common honorific abbreviations.

Used for two things:
1. Optional pre-processing before TTS (the benchmark tests models both with and
   without it, because many models mispronounce raw digits in Malay).
2. Normalising reference and ASR hypothesis before WER/CER, so "25" vs
   "dua puluh lima" is not counted as an error.
"""

from __future__ import annotations

import re

ONES = ["kosong", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan"]
SCALES = [(10**12, "trilion"), (10**9, "bilion"), (10**6, "juta"), (1000, "ribu")]
MONTHS = ["Januari", "Februari", "Mac", "April", "Mei", "Jun", "Julai", "Ogos",
          "September", "Oktober", "November", "Disember"]

ABBREVIATIONS = {
    r"\bDr\.": "Doktor",
    r"\bEn\.": "Encik",
    r"\bPn\.": "Puan",
    r"\bTn\.": "Tuan",
    r"\bCik\.": "Cik",
    r"\bProf\.": "Profesor",
    r"\bdll\b\.?": "dan lain-lain",
    r"\bdsb\b\.?": "dan sebagainya",
    r"\bspt\b\.?": "seperti",
    r"\bkpd\b\.?": "kepada",
    r"\byg\b": "yang",
    r"\bdgn\b": "dengan",
    r"\btdk\b": "tidak",
}


def _below_thousand(n: int) -> str:
    assert 0 <= n < 1000
    words: list[str] = []
    h, rest = divmod(n, 100)
    if h == 1:
        words.append("seratus")
    elif h > 1:
        words += [ONES[h], "ratus"]
    if rest == 0:
        return " ".join(words)
    if rest < 10:
        words.append(ONES[rest])
    elif rest == 10:
        words.append("sepuluh")
    elif rest == 11:
        words.append("sebelas")
    elif rest < 20:
        words += [ONES[rest - 10], "belas"]
    else:
        t, o = divmod(rest, 10)
        words += [ONES[t], "puluh"]
        if o:
            words.append(ONES[o])
    return " ".join(words)


def number_to_words(n: int) -> str:
    """Integer to Malay words. 1000 -> 'seribu', 2026 -> 'dua ribu dua puluh enam'."""
    if n < 0:
        return "negatif " + number_to_words(-n)
    if n == 0:
        return "kosong"
    words: list[str] = []
    for value, name in SCALES:
        q, n = divmod(n, value)
        if q:
            if q == 1 and value == 1000:
                words.append("seribu")
            else:
                words += [number_to_words(q), name]
    if n:
        words.append(_below_thousand(n))
    return " ".join(words)


def digits_to_words(s: str) -> str:
    """Read digit by digit (phone numbers, decimals after 'perpuluhan')."""
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


def ordinal(n: int) -> str:
    return "pertama" if n == 1 else "ke" + number_to_words(n)


def _num(s: str) -> int:
    return int(s.replace(",", ""))


def _decimal(m: re.Match) -> str:
    return f"{number_to_words(_num(m.group(1)))} perpuluhan {digits_to_words(m.group(2))}"


def _currency(m: re.Match) -> str:
    whole = number_to_words(_num(m.group(1)))
    out = f"{whole} ringgit"
    if m.group(2):
        sen = int(m.group(2).ljust(2, "0"))
        if sen:
            out += f" {number_to_words(sen)} sen"
    return out


def _date(m: re.Match) -> str:
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    return f"{number_to_words(d)} {MONTHS[mo - 1]} {number_to_words(y)}"


def _time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return m.group(0)
    out = f"pukul {number_to_words(h)}"
    if mi:
        out += f" {number_to_words(mi)} minit"
    return out


def _phone(m: re.Match) -> str:
    return digits_to_words(m.group(0))


def normalize(text: str, expand_abbreviations: bool = True) -> str:
    t = text
    if expand_abbreviations:
        for pat, rep in ABBREVIATIONS.items():
            t = re.sub(pat, rep, t)
    # phone numbers like 012-345 6789 -> digit by digit
    t = re.sub(r"\b0\d{1,2}-\d{3,4}\s?\d{3,4}\b", _phone, t)
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", _date, t)
    t = re.sub(r"\b(?:[Pp]ukul\s+)?(\d{1,2}):(\d{2})\b", _time, t)
    t = re.sub(r"\bRM\s?(\d[\d,]*)(?:\.(\d{1,2}))?\b", _currency, t)
    t = re.sub(r"(\d[\d,]*)\.(\d+)\s?%", lambda m: _decimal(m) + " peratus", t)
    t = re.sub(r"(\d[\d,]*)\s?%", lambda m: number_to_words(_num(m.group(1))) + " peratus", t)
    t = re.sub(r"\bke-(\d+)\b", lambda m: ordinal(int(m.group(1))), t)
    t = re.sub(r"\b(\d[\d,]*)\.(\d+)\b", _decimal, t)
    t = re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", lambda m: number_to_words(_num(m.group(0))), t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_wer(text: str) -> str:
    """Aggressive normalisation for WER/CER comparison."""
    from .emotion_tags import TAG_RE

    t = TAG_RE.sub(" ", text)
    t = normalize(t).lower()
    t = t.replace("-", " ")  # reduplication: "kanak-kanak" vs "kanak kanak"
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
