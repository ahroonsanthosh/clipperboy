"""Find clip-worthy moments.

Approach: split the transcript into sentences, build every plausible
sentence-aligned window between MIN and MAX clip length, score each window on
signals that correlate with short-form performance, then greedily pick the
best non-overlapping set.

Everything here is local and free. If a local Ollama model is reachable, it is
used only to re-rank and title the shortlist — never required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from backend import config
from backend.pipeline.transcribe import Transcript, Word

# --- lexicons -------------------------------------------------------------

HOOK_STARTERS = (
    "here's the thing", "the truth is", "nobody talks about", "what most people",
    "the biggest mistake", "let me tell you", "i'll never forget", "the crazy part",
    "here's what", "the reason", "most people think", "everyone thinks",
    "i used to", "the first time", "what happened was", "you have to understand",
    "the problem is", "and that's when", "so i said", "imagine",
)

QUESTION_WORDS = ("why", "how", "what if", "do you", "did you", "have you", "what's the")

INSIGHT_WORDS = (
    "realise", "realize", "realised", "realized", "learned", "lesson", "secret",
    "mistake", "truth", "actually", "fundamentally", "the key", "turns out",
    "counterintuitive", "framework", "principle", "rule", "strategy", "advice",
)

EMOTION_WORDS = (
    "insane", "crazy", "incredible", "terrifying", "hilarious", "shocked", "brutal",
    "amazing", "devastating", "obsessed", "love", "hate", "scared", "furious",
    "unbelievable", "wild", "nuts", "beautiful", "painful", "proud",
)

STORY_WORDS = (
    "story", "happened", "remember", "one day", "years ago", "back then",
    "we were", "i was", "suddenly", "then he", "then she", "then they",
)

# Words that make an opening sentence feel like it's mid-thought.
DANGLING_STARTS = (
    "and", "but", "so", "because", "which", "that", "it", "he", "she", "they",
    "them", "this", "those", "also", "then", "or", "yeah", "yes", "no", "well",
)

FILLER = ("um", "uh", "erm", "like", "you know", "i mean", "sort of", "kind of")

AD_MARKERS = (
    "sponsor", "sponsored by", "promo code", "discount code", "this episode is brought",
    "use code", "our partner", "check out our", "patreon", "subscribe to the channel",
    "smash that like", "link in the description", "terms and conditions",
)

STOPWORDS = set("""a an the and or but if then than that this these those i you he she it we they
me him her us them my your his its our their of in on at to for with from by as is are was were be
been being do does did doing have has had having will would can could should may might must not no
so very just really about into over under out up down there here what which who whom when where why
how all any both each few more most other some such only own same too s t don now am
""".split())


@dataclass
class Sentence:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    score: float
    reasons: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


# --- sentence splitting ---------------------------------------------------

_SENT_END = re.compile(r"[.!?]+[\"')\]]*\s*$")


def to_sentences(tr: Transcript) -> list[Sentence]:
    """Whisper segments are already near-sentence sized, but it happily splits
    mid-clause. Merge fragments until we hit terminal punctuation or a long gap."""
    sentences: list[Sentence] = []
    buf_text: list[str] = []
    buf_words: list[Word] = []
    buf_start: float | None = None
    last_end = 0.0

    for seg in tr.segments:
        if buf_start is None:
            buf_start = seg.start
        gap = seg.start - last_end if buf_text else 0.0
        buf_text.append(seg.text.strip())
        buf_words.extend(seg.words)
        last_end = seg.end

        joined = " ".join(buf_text).strip()
        long_enough = (last_end - buf_start) >= 2.0
        if (_SENT_END.search(joined) and long_enough) or gap > 1.2 or (last_end - buf_start) > 18:
            sentences.append(Sentence(buf_start, last_end, joined, list(buf_words)))
            buf_text, buf_words, buf_start = [], [], None

    if buf_text and buf_start is not None:
        sentences.append(Sentence(buf_start, last_end, " ".join(buf_text).strip(), list(buf_words)))
    return [s for s in sentences if s.text]


# --- feature helpers ------------------------------------------------------

def _count_any(text: str, phrases) -> int:
    return sum(1 for p in phrases if p in text)


def _envelope_stats(env: np.ndarray, hz: int, start: float, end: float) -> tuple[float, float, float]:
    """(mean loudness, variability, peak) for a time range, all 0-1ish."""
    if env.size == 0:
        return 0.0, 0.0, 0.0
    a, b = int(start * hz), int(end * hz)
    a, b = max(0, a), min(env.size, max(a + 1, b))
    window = env[a:b]
    if window.size == 0:
        return 0.0, 0.0, 0.0
    ref = float(np.percentile(env, 95)) or 1.0
    return (
        float(window.mean() / ref),
        float(window.std() / ref),
        float(window.max() / ref),
    )


def keywords_for(text: str, limit: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for raw in re.findall(r"[a-zA-Z']{3,}", text.lower()):
        if raw in STOPWORDS:
            continue
        counts[raw] = counts.get(raw, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:limit]]


# --- scoring --------------------------------------------------------------

def score_window(sents: list[Sentence], env: np.ndarray, hz: int,
                 total_duration: float) -> Candidate:
    start, end = sents[0].start, sents[-1].end
    text = " ".join(s.text for s in sents).strip()
    low = text.lower()
    words = [w for s in sents for w in s.words]
    duration = max(0.001, end - start)
    word_count = len(words) or len(text.split())

    opening = sents[0].text.lower().strip()
    first_word = re.sub(r"[^a-z']", "", opening.split(" ")[0]) if opening else ""

    hook = _count_any(opening[:90], HOOK_STARTERS) * 2 + _count_any(low[:160], HOOK_STARTERS)
    questions = text.count("?") + _count_any(low, QUESTION_WORDS)
    insight = _count_any(low, INSIGHT_WORDS)
    emotion = _count_any(low, EMOTION_WORDS)
    story = _count_any(low, STORY_WORDS)
    filler = _count_any(low, FILLER)
    ads = _count_any(low, AD_MARKERS)
    numbers = len(re.findall(r"\b\d[\d,.]*\b", text))

    loud_mean, loud_var, loud_peak = _envelope_stats(env, hz, start, end)
    wpm = (word_count / duration) * 60 if duration else 0

    # Self-containment: does it open cleanly and close on a full stop?
    opens_clean = 0.0 if first_word in DANGLING_STARTS else 1.0
    closes_clean = 1.0 if _SENT_END.search(text) else 0.0

    # Position: intros and outros are usually housekeeping.
    pos = start / total_duration if total_duration else 0.5
    position_penalty = 0.0
    if pos < 0.03:
        position_penalty = 1.5
    elif pos > 0.97:
        position_penalty = 1.0

    # Length sweet spot for short-form: ~30-45s.
    ideal = 38.0
    length_fit = max(0.0, 1.0 - abs(duration - ideal) / ideal)

    density = min(1.5, word_count / max(1.0, duration) / 2.6)  # normalised speech density

    score = (
        2.6 * min(hook, 3)
        + 1.5 * min(questions, 3)
        + 1.6 * min(insight, 4)
        + 1.7 * min(emotion, 4)
        + 1.2 * min(story, 4)
        + 0.7 * min(numbers, 3)
        + 2.2 * opens_clean
        + 1.2 * closes_clean
        + 2.0 * length_fit
        + 2.4 * loud_var          # dynamic delivery / laughter / emphasis
        + 1.2 * loud_peak
        + 1.0 * density
        - 0.6 * min(filler, 6)
        - 4.0 * min(ads, 3)
        - position_penalty
    )
    if word_count < 25:
        score -= 3.0  # too thin to say anything
    if wpm < 70:
        score -= 1.5  # long silences

    reasons: list[str] = []
    if hook:
        reasons.append("Opens with a strong hook")
    if questions:
        reasons.append("Poses a question that pulls the viewer in")
    if insight:
        reasons.append("Contains a concrete insight or lesson")
    if emotion:
        reasons.append("Emotionally charged language")
    if story:
        reasons.append("Tells a self-contained story")
    if loud_var > 0.28:
        reasons.append("Lively delivery — laughter or vocal emphasis")
    if numbers:
        reasons.append("Cites specific numbers or facts")
    if opens_clean and closes_clean:
        reasons.append("Starts and ends on complete thoughts")
    if ads:
        reasons.append("⚠ Looks like sponsor read")
    if not reasons:
        reasons.append("Steady, quotable stretch of conversation")

    return Candidate(
        start=start, end=end, text=text, score=score, reasons=reasons[:4],
        features={
            "hook": hook, "questions": questions, "insight": insight,
            "emotion": emotion, "story": story, "filler": filler,
            "loudness": round(loud_mean, 3), "energy_variation": round(loud_var, 3),
            "words_per_minute": round(wpm), "word_count": word_count,
            "self_contained": bool(opens_clean and closes_clean),
        },
        keywords=keywords_for(text),
    )


def build_candidates(sents: list[Sentence], env: np.ndarray, hz: int,
                     total_duration: float) -> list[Candidate]:
    out: list[Candidate] = []
    n = len(sents)
    for i in range(n):
        for j in range(i, n):
            dur = sents[j].end - sents[i].start
            if dur < config.MIN_CLIP_SEC:
                continue
            if dur > config.MAX_CLIP_SEC:
                break
            out.append(score_window(sents[i:j + 1], env, hz, total_duration))
    return out


def select(candidates: list[Candidate], count: int, min_gap: float = 5.0) -> list[Candidate]:
    """Greedy non-overlapping selection, best score first."""
    chosen: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: -c.score):
        if len(chosen) >= count:
            break
        if any(cand.start < c.end + min_gap and c.start < cand.end + min_gap for c in chosen):
            continue
        chosen.append(cand)
    chosen.sort(key=lambda c: c.start)
    return chosen


# --- titles & summaries ---------------------------------------------------

def _trim_title(text: str, limit: int = 58) -> str:
    text = re.sub(r"\s+", " ", text).strip().strip("-—,;: ")
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(".,;:-— ") + "…"


def local_title_and_summary(cand: Candidate) -> tuple[str, str]:
    """Deterministic, offline title/summary from the clip's own words."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cand.text) if s.strip()]
    if not sentences:
        return "Untitled clip", cand.text[:160]

    def punch(s: str) -> float:
        low = s.lower()
        return (
            2.0 * _count_any(low, HOOK_STARTERS)
            + 1.5 * _count_any(low, INSIGHT_WORDS)
            + 1.5 * _count_any(low, EMOTION_WORDS)
            + 1.0 * s.count("?")
            + 0.02 * len(s)
            - (2.0 if len(s.split()) < 5 else 0.0)
        )

    best = max(sentences, key=punch)
    title = _trim_title(best)
    summary = " ".join(sentences[:2])
    if len(summary) > 220:
        summary = summary[:217].rsplit(" ", 1)[0] + "…"
    return title, summary


def enrich(cands: list[Candidate], progress=None) -> list[Candidate]:
    for c in cands:
        c.title, c.summary = local_title_and_summary(c)
    if config.USE_OLLAMA != "off":
        try:
            _ollama_enrich(cands, progress)
        except Exception:
            pass  # optional by design — local titles already in place
    return cands


def _ollama_enrich(cands: list[Candidate], progress=None) -> None:
    import httpx

    with httpx.Client(base_url=config.OLLAMA_HOST, timeout=8.0) as probe_client:
        probe_client.get("/api/tags").raise_for_status()

    if progress:
        progress("Polishing titles with local model")

    with httpx.Client(base_url=config.OLLAMA_HOST, timeout=90.0) as client:
        for c in cands:
            prompt = (
                "You write titles for social media clips. Given the transcript below, reply with "
                "exactly two lines and nothing else:\n"
                "TITLE: a punchy hook under 60 characters\n"
                "SUMMARY: one sentence describing what is discussed\n\n"
                f"Transcript:\n{c.text[:2200]}"
            )
            resp = client.post("/api/generate", json={
                "model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.4},
            })
            resp.raise_for_status()
            body = resp.json().get("response", "")
            title = summary = ""
            for line in body.splitlines():
                if line.upper().startswith("TITLE:"):
                    title = line.split(":", 1)[1].strip().strip('"')
                elif line.upper().startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip().strip('"')
            if title:
                c.title = _trim_title(title)
            if summary:
                c.summary = summary


def find_clips(tr: Transcript, env: np.ndarray, hz: int, count: int,
               progress=None) -> list[Candidate]:
    sents = to_sentences(tr)
    if not sents:
        return []
    cands = build_candidates(sents, env, hz, tr.duration)
    if not cands:
        # Podcast shorter than one minimum-length window: use the whole thing.
        cands = [score_window(sents, env, hz, tr.duration)]
    picked = select(cands, count)
    return enrich(picked, progress)
