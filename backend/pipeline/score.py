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
    "who", "whose", "however", "therefore", "anyway", "besides", "plus", "meaning",
    "again", "ok", "okay", "alright", "right", "um", "uh", "erm", "though", "although",
    "otherwise", "thus", "hence", "him", "her", "his", "its", "their", "these",
)

# Openings that explicitly point *backwards* at something the viewer never saw.
BACKREF_STARTS = (
    "as i was saying", "like i said", "like i mentioned", "as i mentioned",
    "as we said", "going back to", "back to what", "to that point", "on that note",
    "that's why", "that is why", "which is why", "which means", "so anyway",
    "so yeah", "so that", "and then", "and so", "but the thing is", "in that case",
    "the other one", "the second one", "the second thing", "the third thing",
    "same thing", "exactly", "for that reason", "either way", "speaking of",
)

# Words a *closing* sentence must never end on — the thought clearly continues.
DANGLING_ENDS = (
    "and", "but", "or", "so", "because", "which", "that", "if", "when", "while",
    "with", "to", "for", "of", "the", "a", "an", "my", "your", "his", "her", "their",
    "our", "its", "is", "was", "are", "were", "than", "then", "as", "at", "in", "on",
    "like", "plus", "however", "therefore", "about", "into", "from", "by", "not",
    "very", "just", "really", "i", "we", "he", "she", "they", "it", "you", "there",
    "how", "why", "what", "who", "some", "any", "one", "two", "three", "first",
    "second", "third", "another", "also", "um", "uh", "erm",
)

# Phrases that announce more to come: a clip ending here is obviously truncated.
DANGLING_END_PHRASES = (
    "the second thing is", "the second thing", "the third thing", "the other thing is",
    "the first thing is", "first of all", "for example", "such as", "one of the",
    "number one", "number two", "the first is", "the second is", "which means",
    "the reason is", "here's why", "here is why", "let me explain", "let me tell you",
    "here's the thing", "here is the thing", "the point is", "and it goes like this",
    "listen to this", "check this out", "watch this", "for instance", "like this",
    "as follows", "the problem is", "the truth is", "here's what",
)

# A clip that ends on a direct question nobody answered feels cut off.
_OPEN_QUESTION_STARTERS = (
    "why", "how", "what", "when", "where", "who", "do", "did", "does", "is", "are",
    "was", "were", "can", "could", "would", "should", "have", "has", "will", "am",
)

# Silence long enough to be believable evidence of a thought boundary, even when
# Whisper's punctuation is missing or wrong.
GAP_STRONG = 0.35      # seconds of silence that confirm a real boundary
GAP_SNAP_MAX = 0.35    # never pad a clip by more than this on either side

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
    # Index range into the sentence list, so the final cut can be snapped into
    # the surrounding silence without re-deriving it from timestamps.
    sent_range: tuple[int, int] | None = None

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
        # A long pause ends the *previous* sentence — flush before absorbing
        # this segment, or the new thought gets glued onto the old one.
        if buf_text and buf_start is not None and (seg.start - last_end) > 1.2:
            sentences.append(Sentence(buf_start, last_end, " ".join(buf_text).strip(),
                                      list(buf_words)))
            buf_text, buf_words, buf_start = [], [], None

        if buf_start is None:
            buf_start = seg.start
        buf_text.append(seg.text.strip())
        buf_words.extend(seg.words)
        last_end = seg.end

        joined = " ".join(buf_text).strip()
        long_enough = (last_end - buf_start) >= 2.0
        if (_SENT_END.search(joined) and long_enough) or (last_end - buf_start) > 18:
            sentences.append(Sentence(buf_start, last_end, joined, list(buf_words)))
            buf_text, buf_words, buf_start = [], [], None

    if buf_text and buf_start is not None:
        sentences.append(Sentence(buf_start, last_end, " ".join(buf_text).strip(), list(buf_words)))
    return [s for s in sentences if s.text]


# --- boundary quality -----------------------------------------------------
#
# The user's rule: a clip must be a COMPLETE THOUGHT. So boundary quality is a
# hard filter applied to the (i, j) index pair *before* any window is scored —
# rejected windows cost a couple of array lookups instead of a full score pass.
#
# Every sentence gets two grades, computed once per episode in O(n):
#
#   2 = clean   — punctuation *and/or* a real silence gap agree this is a
#                 genuine thought boundary
#   1 = usable  — nothing says it's mid-sentence, but nothing confirms it either
#                 (sloppy punctuation, no pause)
#   0 = broken  — positively mid-thought: dangling word, back-reference,
#                 trailing conjunction, unanswered question
#
# A candidate's tier is min(start_grade[i], end_grade[j]).

TIER_CLEAN = 2
TIER_USABLE = 1
TIER_BROKEN = 0

_BOUNDARY_QUALITY = {TIER_CLEAN: "clean", TIER_USABLE: "approximate", TIER_BROKEN: "forced"}

_WORD_RE = re.compile(r"[a-z']+")


def _first_token(text: str) -> str:
    m = _WORD_RE.search(text.lower())
    return m.group(0) if m else ""


def _last_token(text: str) -> str:
    toks = _WORD_RE.findall(text.lower())
    return toks[-1] if toks else ""


def _tail(text: str, words: int = 7) -> str:
    """The last few words, lowercased — used for 'about to continue' phrases."""
    return " ".join(text.lower().split()[-words:])


def _starts_mid_thought(text: str) -> bool:
    low = text.lower().lstrip("\"'([ ")
    if _first_token(low) in DANGLING_STARTS:
        return True
    return any(low.startswith(p) for p in BACKREF_STARTS)


def _ends_mid_thought(text: str) -> bool:
    stripped = text.rstrip().rstrip("\"')]")
    if _last_token(stripped) in DANGLING_ENDS:
        return True
    tail = _tail(stripped)
    if any(tail.endswith(p) or tail.endswith(p + ",") for p in DANGLING_END_PHRASES):
        return True
    # An ellipsis or a trailing comma/semicolon is an explicit "more follows".
    if stripped.endswith(("...", "…", ",", ";", ":", "-", "—")):
        return True
    # A direct question with no answer inside the clip.
    if stripped.endswith("?"):
        last_q = re.split(r"(?<=[.!?])\s+", stripped)[-1]
        if _first_token(last_q) in _OPEN_QUESTION_STARTERS:
            return True
    return False


def boundary_grades(sents: list[Sentence]) -> tuple[list[int], list[int]]:
    """Per-sentence (start_grade, end_grade). O(n), computed once per episode."""
    n = len(sents)
    starts = [TIER_USABLE] * n
    ends = [TIER_USABLE] * n
    if n == 0:
        return starts, ends

    for i, s in enumerate(sents):
        text = s.text.strip()

        if _starts_mid_thought(text):
            starts[i] = TIER_BROKEN
        else:
            gap_before = s.start - sents[i - 1].end if i else float("inf")
            prev_closed = i == 0 or bool(_SENT_END.search(sents[i - 1].text))
            # Punctuation OR silence is enough; Whisper rarely gets both wrong.
            if prev_closed or gap_before >= GAP_STRONG:
                starts[i] = TIER_CLEAN

        if _ends_mid_thought(text):
            ends[i] = TIER_BROKEN
        else:
            gap_after = sents[i + 1].start - s.end if i + 1 < n else float("inf")
            if _SENT_END.search(text) or gap_after >= GAP_STRONG:
                ends[i] = TIER_CLEAN

    return starts, ends


def silence_gaps(sents: list[Sentence]) -> tuple[list[float], list[float]]:
    """Silence before / after each sentence, in seconds."""
    n = len(sents)
    before = [float("inf")] * n
    after = [float("inf")] * n
    for i in range(n):
        if i:
            before[i] = max(0.0, sents[i].start - sents[i - 1].end)
        if i + 1 < n:
            after[i] = max(0.0, sents[i + 1].start - sents[i].end)
    return before, after


def snap_to_silence(start: float, end: float, gap_before: float, gap_after: float,
                    total_duration: float = 0.0,
                    max_pad: float = GAP_SNAP_MAX) -> tuple[float, float]:
    """Move the cut into the middle of the surrounding silence.

    Cutting exactly on the first word clips the breath before it and lands on a
    hard consonant; half a beat of room makes the clip sound intentional. Padding
    is capped so the clip cannot outgrow MAX_CLIP_SEC.
    """
    pad_head = min(max_pad, gap_before / 2.0) if gap_before != float("inf") else max_pad
    pad_tail = min(max_pad, gap_after / 2.0) if gap_after != float("inf") else max_pad
    pad_head = max(0.0, pad_head)
    pad_tail = max(0.0, pad_tail)

    head_room = config.MAX_CLIP_SEC - (end - start)
    if head_room <= 0:
        return start, end
    if pad_head + pad_tail > head_room:  # share what's left proportionally
        total = pad_head + pad_tail
        pad_head *= head_room / total
        pad_tail *= head_room / total

    new_start = max(0.0, start - pad_head)
    new_end = end + pad_tail
    if total_duration:
        new_end = min(new_end, total_duration)
    return new_start, max(new_start + 0.001, new_end)


# --- feature helpers ------------------------------------------------------

def _count_any(text: str, phrases) -> int:
    return sum(1 for p in phrases if p in text)


def loudness_reference(env: np.ndarray) -> float:
    """The 95th-percentile loudness, used to normalise every window.

    Hoisted out of the per-window scorer: it is constant for an episode, and
    recomputing it for each of ~10,000 candidates dominated the analysis stage.
    """
    if env.size == 0:
        return 1.0
    return float(np.percentile(env, 95)) or 1.0


def _envelope_stats(env: np.ndarray, hz: int, start: float, end: float,
                    ref: float | None = None) -> tuple[float, float, float]:
    """(mean loudness, variability, peak) for a time range, all 0-1ish."""
    if env.size == 0:
        return 0.0, 0.0, 0.0
    a, b = int(start * hz), int(end * hz)
    a, b = max(0, a), min(env.size, max(a + 1, b))
    window = env[a:b]
    if window.size == 0:
        return 0.0, 0.0, 0.0
    ref = loudness_reference(env) if ref is None else ref
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
                 total_duration: float, ref: float | None = None,
                 tier: int = TIER_CLEAN) -> Candidate:
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

    loud_mean, loud_var, loud_peak = _envelope_stats(env, hz, start, end, ref)
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
    if tier < TIER_CLEAN:
        reasons.append("⚠ Boundaries are approximate — check the in/out points")
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
            "boundary_tier": tier,
            "boundary_quality": _BOUNDARY_QUALITY.get(tier, "approximate"),
            "complete_thought": tier >= TIER_CLEAN,
        },
        keywords=keywords_for(text),
    )


def candidate_spans(sents: list[Sentence], tier: int) -> list[tuple[int, int, int]]:
    """Sentence-index pairs whose boundary tier is >= `tier` and length is legal.

    Deliberately does no scoring: rejecting a window here costs two list lookups,
    where scoring it costs a full feature extraction. On a 60-minute episode the
    strict pass throws away most (i, j) pairs before they are ever scored.
    """
    starts, ends = boundary_grades(sents)
    n = len(sents)
    spans: list[tuple[int, int, int]] = []
    min_sec, max_sec = config.MIN_CLIP_SEC, config.MAX_CLIP_SEC
    for i in range(n):
        sg = starts[i]
        if sg < tier:
            continue  # whole row is unusable — skip without touching durations
        s_start = sents[i].start
        for j in range(i, n):
            dur = sents[j].end - s_start
            if dur < min_sec:
                continue
            if dur > max_sec:
                break
            eg = ends[j]
            if eg >= tier:
                spans.append((i, j, sg if sg < eg else eg))
    return spans


def build_candidates(sents: list[Sentence], env: np.ndarray, hz: int,
                     total_duration: float, progress=None,
                     tier: int | None = None) -> list[Candidate]:
    """Score every window that survives the boundary constraints.

    With `tier=None` (the default) the strictest tier that still yields any
    window wins: clean boundaries if the transcript has them, otherwise
    "nothing says this is mid-sentence", otherwise — for a transcript with no
    punctuation and no pauses at all — whatever exists. Callers that want to
    top up a thin strict selection ask for a lower tier explicitly.
    """
    n = len(sents)
    if n == 0:
        return []

    tiers = (tier,) if tier is not None else (TIER_CLEAN, TIER_USABLE, TIER_BROKEN)
    spans: list[tuple[int, int, int]] = []
    for t in tiers:
        spans = candidate_spans(sents, t)
        if spans:
            break
    if not spans:
        return []

    ref = loudness_reference(env)
    out: list[Candidate] = []
    total = len(spans)
    step = max(1, total // 40)
    for k, (i, j, t) in enumerate(spans):
        if progress and total > 200 and k % step == 0:
            progress(k / total, f"Scoring moments… {int(k / total * 100)}%")
        cand = score_window(sents[i:j + 1], env, hz, total_duration, ref, t)
        cand.sent_range = (i, j)
        out.append(cand)
    return out


def select(candidates: list[Candidate], count: int, min_gap: float = 5.0,
           seed: list[Candidate] | None = None) -> list[Candidate]:
    """Greedy non-overlapping selection, best score first.

    `seed` is an already-chosen set (e.g. the strict-boundary picks) that new
    candidates must not overlap — used to top up a thin strict selection with
    lower-confidence clips without disturbing the good ones.
    """
    chosen: list[Candidate] = list(seed or [])
    for cand in sorted(candidates, key=lambda c: -c.score):
        if any(cand is c for c in chosen):
            continue
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


def apply_snapping(picked: list[Candidate], sents: list[Sentence],
                   total_duration: float = 0.0, min_gap: float = 0.0) -> list[Candidate]:
    """Move each chosen clip's in/out points into the surrounding silence."""
    if not picked:
        return picked
    before, after = silence_gaps(sents)
    picked = sorted(picked, key=lambda c: c.start)
    for idx, c in enumerate(picked):
        if c.sent_range is None:
            continue
        i, j = c.sent_range
        if not (0 <= i < len(sents) and 0 <= j < len(sents)):
            continue
        start, end = snap_to_silence(c.start, c.end, before[i], after[j], total_duration)
        # Never let padding eat into a neighbouring clip.
        if idx and start < picked[idx - 1].end + min_gap:
            start = c.start
        if idx + 1 < len(picked) and end > picked[idx + 1].start - min_gap:
            end = c.end
        c.start, c.end = start, end
    return picked


def find_clips(tr: Transcript, env: np.ndarray, hz: int, count: int,
               progress=None, scan_progress=None) -> list[Candidate]:
    sents = to_sentences(tr)
    if not sents:
        return []

    # Pass 1: only windows that begin and end on a genuine thought boundary.
    cands = build_candidates(sents, env, hz, tr.duration, progress=scan_progress)
    picked = select(cands, count)

    # Graceful degradation: a transcript with sloppy punctuation and no pauses
    # can starve the strict pass. Rather than return two clips, top up from the
    # next tier down — those arrive flagged (features["boundary_quality"]) so
    # the UI can warn that the in/out points are approximate.
    if len(picked) < count:
        best_tier = min((c.features.get("boundary_tier", TIER_BROKEN) for c in cands),
                        default=TIER_CLEAN)
        for lower in (TIER_USABLE, TIER_BROKEN):
            if len(picked) >= count or lower >= best_tier:
                continue
            relaxed = build_candidates(sents, env, hz, tr.duration, tier=lower)
            picked = select(relaxed, count, seed=picked)

    if not picked:
        # Shorter than one minimum-length window: use the whole thing.
        whole = score_window(sents, env, hz, tr.duration, tier=TIER_USABLE)
        whole.sent_range = (0, len(sents) - 1)
        picked = [whole]

    picked = apply_snapping(picked, sents, tr.duration)
    picked.sort(key=lambda c: c.start)
    return enrich(picked, progress)
