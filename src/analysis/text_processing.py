"""Text processing: sentence splitting, scoring, summarization.

Extracted from cli.py to improve modularity and testability.
"""

from __future__ import annotations

import re

# Default patterns — overridable via config
_FILLER_PATTERNS = [
    r"^here (?:are|is) \w+",
    r"^(?:but |and |so |yet )",
    r"^over the (?:past|last) \w+",
    r"^in recent (?:years|months)",
    r"^(?:this|that) (?:comes?|came)",
    r"never been (?:easier|harder)",
    r"^in \d{4},?\s",
    r"^(?:back )?in the \d{4}s",
    r"^more than (?:the|a) ",
    r"^after (?:finishing|completing|graduating)",
    r"^(?:he|she|they) (?:was|were) (?:born|raised|assigned)",
    r"^the \d+-year-old",
    r"^(?:one|two|three) of .{0,20}(?:most|first|last)",
]

_KEY_POINT_PATTERNS = [
    r"(?:will|would|may|could|should) (?:continue|remain|face|see|lead|result)",
    r"(?:is|are) (?:expected|likely|set|poised|preparing) to",
    r"(?:announced?|unveiled?|revealed?|confirmed?) (?:that|plans?|a new)",
    r"(?:according to|said|stated|noted|emphasized)",
    r"(?:the|this) (?:move|decision|policy|measure|action) (?:will|would|could|may)",
    r"(?:signals?|indicates?|suggests?|shows?|reflects?) (?:that|a |the )",
    r"^(?:china|beijing|the (?:u\.?s\.?|us)|washington|canada|ottawa)",
]

_BOILERPLATE_PATTERNS = [
    r"(?:our |the )?privacy (?:statement|policy|notice)",
    r"cookie (?:policy|notice|consent)",
    r"by continuing to (?:browse|use|visit)",
    r"you agree to (?:our |the )?use of cookies",
    r"revised privacy policy",
    r"terms of (?:use|service)",
    r"choose your language",
    r"select (?:your )?language",
    r"subscribe to (?:our )?newsletter",
    r"sign up for (?:our )?newsletter",
    r"follow us on",
    r"share this (?:article|story)",
    r"互联网新闻信息(?:服务)?许可证",
    r"disinformation report hotline",
    r"举报电话",
    r"备案号",
    r"ICP备",
    r"(?:©|copyright|\(c\))\s*\d{4}",
    r"all rights reserved",
    r"click here to",
    r"read more:",
    r"related (?:articles?|stories?|news):",
    # News site membership/promotion boilerplate
    r"become a member",
    r"support (?:our |independent )?journalism",
    r"(?:free|premium) membership",
    r"(?:monthly|annual) subscription",
    r"donate (?:now|today)",
    r"join (?:our )?community",
    r"(?:hkfp|scmp|rthk|cna) (?:keychain|tote|bag|merchandise)",
    r"support hkfp",
    r"fact[- ]check(?:ed)? by",
    # Reporter credits and photo captions
    r"^\s*[（(](?:记者|記者|摄|攝)[^）)]*[）)]",
    r"^\s*〔(?:记者|記者)[^〕]*〕",
    r"^\s*(?:photo|image|图片)\s*(?:credit|courtesy|by)\s*:",
    # Navigation and social elements
    r"(?:previous|next) (?:article|story|post)",
    r"(?:trending|popular|most read) (?:stories|articles|news)",
    r"(?:share|tweet|post) (?:on|to) (?:facebook|twitter|x|whatsapp|linkedin)",
    r"(?:print|email) this (?:article|story)",
    # Chinese news site boilerplate
    r"(?:来源|來源)\s*[：:]\s*\S+",
    r"(?:编辑|編輯|责编|責編)\s*[：:]\s*\S+",
    r"(?:原标题|原標題)\s*[：:]",
    r"(?:转载|轉載)请注明",
]


def clean_body_text(
    text: str,
    boilerplate_patterns: list[str] | None = None,
) -> str:
    """Remove boilerplate and junk from article body text.

    Strips lines matching boilerplate patterns, reporter credits,
    and other non-content text. Applied before body is passed
    to the LLM for perspective generation.
    """
    if not text:
        return text
    patterns = boilerplate_patterns if boilerplate_patterns is not None else _BOILERPLATE_PATTERNS
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_boilerplate = False
        for pattern in patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                is_boilerplate = True
                break
        if not is_boilerplate:
            clean_lines.append(stripped)
    return "\n".join(clean_lines)


def split_sentences(text: str, min_len: int = 15) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z\u201c\u2018\"\'(])', text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > min_len]


def score_sentence(
    sentence: str,
    title: str,
    position: int,
    total: int,
    filler_patterns: list[str] | None = None,
    key_point_patterns: list[str] | None = None,
) -> float:
    """Score a sentence for informativeness."""
    filler = filler_patterns if filler_patterns is not None else _FILLER_PATTERNS
    key_points = key_point_patterns if key_point_patterns is not None else _KEY_POINT_PATTERNS

    s_lower = sentence.lower()
    t_lower = title.lower()
    score = 0.0

    # Numbers and data points
    num_pattern = r'\d+[\d,.]*\s*(?:%|percent|billion|million|thousand|days?|countries)?'
    numbers = re.findall(num_pattern, sentence)
    score += len(numbers) * 2.0

    # Proper nouns
    proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', sentence)
    score += min(len(proper_nouns), 3) * 0.5

    # Title word overlap
    title_words = set(re.findall(r'\b\w{4,}\b', t_lower))
    sent_words = set(re.findall(r'\b\w{4,}\b', s_lower))
    overlap = len(title_words & sent_words)
    score += overlap * 3.0

    if title_words and overlap == 0:
        score -= 2.0

    # Action verbs
    action_pat = r'\b(?:announced?|said|allow|permit|grant|require|impose|launch|sign|ban|approv)'
    if re.search(action_pat, s_lower):
        score += 1.5

    # Key point patterns
    for pat in key_points:
        if re.search(pat, s_lower):
            score += 2.5
            break

    # Filler penalty
    for pat in filler:
        if re.search(pat, s_lower):
            score -= 4.0
            break

    # Length penalties
    if len(sentence) < 60:
        score -= 1.0
    if len(sentence) > 350:
        score -= 1.5

    # Position bias
    if position < 3:
        score += 1.5
    elif 0.3 < position / max(total, 1) < 0.7:
        score -= 0.5

    # Tagged content boost
    if sentence.startswith("[heading] ") or sentence.startswith("[item] "):
        score += 3.0

    return score


def extract_list_items(text: str) -> list[str]:
    """Extract [heading] and [item] tagged lines from enriched body text."""
    items: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("[heading] "):
            items.append(line[10:])
        elif line.startswith("[item] "):
            items.append(line[7:])
    return items


def is_list_headline(title: str) -> bool:
    """Check if headline promises a list (e.g. '5 ways', '3 reasons')."""
    list_pat = r'\b\d+\s+(?:way|reason|thing|tip|step|method|sign|trend|takeaway)'
    return bool(re.search(list_pat, title, re.I))


def remove_boilerplate(
    text: str,
    boilerplate_patterns: list[str] | None = None,
) -> str:
    """Remove website boilerplate text (privacy notices, footers, etc.)."""
    if not text:
        return text

    patterns = boilerplate_patterns if boilerplate_patterns is not None else _BOILERPLATE_PATTERNS
    sentences = split_sentences(text)
    cleaned = []

    for sent in sentences:
        s_lower = sent.lower()
        is_boilerplate = False
        for pattern in patterns:
            if re.search(pattern, s_lower, re.IGNORECASE):
                is_boilerplate = True
                break
        if not is_boilerplate:
            cleaned.append(sent)

    return " ".join(cleaned)


def summarize_body(
    text: str,
    title: str,
    max_chars: int = 500,
    filler_patterns: list[str] | None = None,
    key_point_patterns: list[str] | None = None,
    boilerplate_patterns: list[str] | None = None,
) -> str:
    """Produce an extractive summary from article body text."""
    if not text:
        return ""

    # Clean up ECNS-style artifacts
    text = re.sub(r'^\s*\[heading\]\s*Text:AAAPrint[^\n]*\n*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^ECNS Wire\s*\(ECNS\)\s*[-–—]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Ecns wire\s*\(ECNS\)\s*[-–—]\s*', '', text, flags=re.IGNORECASE)

    # Remove boilerplate
    text = remove_boilerplate(text, boilerplate_patterns)

    # List-style articles
    if is_list_headline(title):
        items = extract_list_items(text)
        if len(items) >= 2:
            summary_parts: list[str] = []
            total = 0
            for item in items:
                short = item.split(".")[0].strip() if len(item) > 100 else item
                if total + len(short) + 4 > max_chars:
                    break
                summary_parts.append(short)
                total += len(short) + 4
            if len(summary_parts) >= 2:
                return " • ".join(summary_parts)

    # Regular articles: extractive summarization
    lines = [ln for ln in text.split("\n") if not ln.strip().startswith(("[heading]", "[item]"))]
    clean = " ".join(lines)
    sentences = split_sentences(clean)
    if not sentences:
        return ""

    # Score and rank
    scored: list[tuple[int, float, str]] = []
    for i, sent in enumerate(sentences):
        sc = score_sentence(sent, title, i, len(sentences), filler_patterns, key_point_patterns)
        scored.append((i, sc, sent))

    by_score = sorted(scored, key=lambda x: -x[1])

    # Select top sentences within budget
    selected: list[int] = []
    total_len = 0
    for idx, _sc, sent in by_score:
        added = len(sent) + (1 if total_len else 0)
        if total_len > 0 and total_len + added > max_chars:
            continue
        selected.append(idx)
        total_len += added
        if total_len >= max_chars * 0.75:
            break

    if not selected:
        return sentences[0] if sentences else ""

    selected.sort()

    # Coherence check
    if selected and selected[0] > 2:
        lede_summary = ""
        for sent in sentences[:3]:
            if len(lede_summary) + len(sent) + 1 <= max_chars:
                lede_summary += (" " if lede_summary else "") + sent
            else:
                break
        if lede_summary:
            return ensure_complete_sentences(lede_summary)

    summary = " ".join(sentences[i] for i in selected)
    return ensure_complete_sentences(summary)


def ensure_complete_sentences(text: str) -> str:
    """Ensure text ends with complete sentences."""
    if not text:
        return text

    text = text.rstrip()

    if text[-1] in ".!?。！？":
        return text

    last_punct = max(
        text.rfind("."),
        text.rfind("!"),
        text.rfind("?"),
        text.rfind("。"),
        text.rfind("！"),
        text.rfind("？"),
    )

    if last_punct > 20:
        return text[: last_punct + 1]

    return text + "..."
