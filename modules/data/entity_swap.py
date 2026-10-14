"""Entity types for answers, same-type substitutes, and the counterfactual bookkeeping built on them.

The counterfactual condition asks one question: when the evidence says something different from
what the model learned, does the model repeat the evidence or its memory? Answering it needs three
things that need no model and no new dependency, and they live here so a test can run them on a
laptop:

1. **What kind of thing is the answer.** ``answer_type`` reads the question's wh-form and head noun
   and the answer's surface shape (digits, month names, capitalization). It is a heuristic by
   design: a gazetteer built from the corpus's own answers plus a few hundred words of cue lists,
   not an NER model. It is wrong in a known way for a "who" question answered by a team, and the
   cost of that is a swap that reads oddly, not a wrong label anywhere that matters.
2. **A same-type substitute.** ``Gazetteer`` holds the distinct answers of each type, optionally
   split by a variant (the asking head noun for places and organizations, the unit for numbers,
   whether a date carries a year), so a city is swapped for a city and "5 km" for another distance
   rather than for "3 years". ``draw`` never returns anything the caller excluded, that contains or
   is contained in an excluded string, or that already occurs in the text the substitute is going
   into.
3. **The swap itself**, on word boundaries, every occurrence, counted.

The second half of the file is the counterfactual record bookkeeping (eligibility, the swapped
evidence row, strata and the memorization ratio), kept free of the model so the arithmetic is
testable without a GPU. ``scripts/eval_abstention.py`` supplies the generation and the
log-probabilities.

Pure Python: ``re``, ``math``, ``random`` and ``collections`` only.
"""
import math
import random
import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ENTITY_TYPES = ("DATE", "YEAR", "NUMBER", "PERSON", "LOCATION", "ORGANIZATION", "PROPER", "COMMON")
SWAPPABLE = ("DATE", "YEAR", "NUMBER", "PERSON", "LOCATION", "ORGANIZATION", "PROPER")

# a variant pool smaller than this is not worth drawing from: the type pool is used instead
MIN_VARIANT_POOL = 20

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
_DATE_PATTERNS = [
    re.compile(rf"^{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{3,4}})?$", re.I),
    re.compile(rf"^\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}(?:,?\s+\d{{3,4}})?$", re.I),
    re.compile(rf"^{_MONTH},?\s+\d{{3,4}}$", re.I),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
]
_YEAR_PATTERN = re.compile(
    r"^(?:(?:in|by|c\.?|circa|after|before|since|until|around|about|late|early|mid)\s+)?(\d{4})s?$",
    re.I,
)
_NUMBER_PATTERN = re.compile(
    r"^[\$£€]?\d[\d,]*(?:\.\d+)?(?:%|(?:st|nd|rd|th))?(?P<unit>(?:\s+[a-z%][\w.\-%]*){0,3})$"
)
_NUMBER_WORDS = frozenset((
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
    "hundred thousand million billion trillion dozen"
).split())
_UNIT_STOPWORDS = frozenset(
    "of the in on and to at for by with from as is was were are a an or that which who".split()
)
_CONNECTORS = frozenset(
    "of the de la le von van der den du da di del al bin ibn and & for in on at y e el".split()
)

_PERSON_NOUNS = frozenset((
    "person man woman actor actress author writer singer president king queen emperor scientist "
    "physicist composer director artist painter poet leader general coach player prime minister "
    "founder inventor philosopher architect explorer".split()
))
_LOCATION_NOUNS = frozenset((
    "city country state county river mountain region island town continent capital province "
    "village lake ocean sea desert area territory district"
).split())
_ORGANIZATION_NOUNS = frozenset((
    "company team university college school band party organization organisation newspaper "
    "airline club label corporation agency studio network league court department committee "
    "council institution"
).split())
_ORG_WORDS = frozenset((
    "inc inc. corp corp. corporation company co co. ltd university college institute bank "
    "association society committee council army navy league party church group club foundation "
    "airlines records studios press department ministry bureau agency commission office authority "
    "board court government parliament senate congress"
).split())
_LOCATION_WORDS = frozenset((
    "river island islands ocean sea lake county republic kingdom empire province peninsula bay gulf "
    "desert valley canyon mountains mount"
).split())

_HEAD_NOUNS = sorted(_PERSON_NOUNS | _LOCATION_NOUNS | _ORGANIZATION_NOUNS, key=len, reverse=True)
_HEAD_PATTERN = re.compile(
    r"\b(?:what|which)\s+(?:[\w'\-]+\s+){0,2}?(" + "|".join(_HEAD_NOUNS) + r")\b"
)
_PROPER_HEAD_PATTERN = re.compile(
    r"\b(?:what|which)\s+(?:(?:is|was|are|were|did|does|do|the|a|an|its|their|his|her)\s+)*([a-z][\w\-]*)"
)
# first words after what or which that say nothing about the kind of name asked for
_VAGUE_HEADS = frozenset(
    "name kind type sort one other famous main first last new old only most least major".split()
)
# kinds of answer where a draw from the wider type pool reads wrongly, so a variant pool that is
# too small means no substitute rather than a fallback
_STRICT_VARIANT_TYPES = frozenset(("NUMBER", "PROPER"))
_WHO_PATTERN = re.compile(r"\bwho(?:m|se)?\b")
_WHERE_PATTERN = re.compile(r"\bwhere\b")


def normalize(text: str) -> str:
    """SQuAD answer normalization: lowercase, no punctuation, no articles, collapsed whitespace.

    The same rule as ``scripts/eval_abstention.normalize_answer``, restated because nothing under
    ``modules/`` may import a script.

    Args:
        text: any string.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(t for t in text.split() if t not in ("a", "an", "the"))


def asking_head(question: str) -> str:
    """The head noun a "what/which <noun>" question asks for, from the known noun lists, or "".

    Args:
        question: the question text.
    """
    match = _HEAD_PATTERN.search(question.lower())
    return match.group(1) if match else ""


def _proper_core(answer: str) -> Optional[List[str]]:
    """The answer's words when it reads as a proper name, else None.

    A leading "the" is dropped; lowercase connectors ("of", "de", "and") are allowed between
    capitalized words; every other word must start with a capital letter or a digit.
    """
    words = answer.split()
    if words and words[0].lower() == "the":
        words = words[1:]
    if not words or len(words) > 8:
        return None
    capitalized = 0
    for i, word in enumerate(words):
        bare = word.strip("\"'()[]")
        if not bare:
            return None
        if bare[0].isupper() or bare[0].isdigit():
            capitalized += 1
        elif bare.lower() in _CONNECTORS and 0 < i < len(words) - 1:
            continue
        else:
            return None
    return words if capitalized else None


def _is_number(answer: str) -> Optional[str]:
    """The unit tail (possibly "") when the answer is a number with an optional unit, else None."""
    match = _NUMBER_PATTERN.match(answer)
    if match:
        unit = match.group("unit").split()
        return None if any(u in _UNIT_STOPWORDS for u in unit) else " ".join(unit)
    tokens = [t for t in re.split(r"[\s\-]+", answer) if t]
    consumed = 0
    while consumed < len(tokens) and (
        tokens[consumed].lower() in _NUMBER_WORDS
        or (tokens[consumed].lower() == "and" and 0 < consumed < len(tokens) - 1)
    ):
        consumed += 1
    if consumed == 0:
        return None
    unit = tokens[consumed:]
    if len(unit) > 3 or any(not u[0].islower() or u.lower() in _UNIT_STOPWORDS for u in unit):
        return None
    return " ".join(unit)


def answer_type(question: str, answer: str) -> str:
    """One of ``ENTITY_TYPES`` for a (question, answer) pair.

    The answer's own surface decides dates, years and numbers; for names the question's wh-form and
    head noun decide first and the answer's words (a company suffix, a river) only break ties.
    Anything that is not a proper name or a number is ``COMMON`` and not swappable.

    Args:
        question: the question text.
        answer: the reference answer.
    """
    answer = answer.strip()
    if not answer:
        return "COMMON"
    if any(p.match(answer) for p in _DATE_PATTERNS):
        return "DATE"
    year = _YEAR_PATTERN.match(answer)
    if year and 1000 <= int(year.group(1)) <= 2099:
        return "YEAR"
    if _is_number(answer) is not None:
        return "NUMBER"
    core = _proper_core(answer)
    if core is None:
        return "COMMON"
    lowered_words = {w.lower().strip(".,") for w in core}
    head = asking_head(question)
    q = question.lower()
    org_word = bool(lowered_words & _ORG_WORDS)
    if any(w.strip(",.").isdigit() for w in core):
        # "Hong Kong in 1894" is a place and a date at once; a swap of the whole string is neither
        return "PROPER"
    if head in _ORGANIZATION_NOUNS:
        return "ORGANIZATION"
    if head in _LOCATION_NOUNS:
        return "LOCATION"
    if head in _PERSON_NOUNS:
        return "PERSON" if len(core) <= 4 else "PROPER"
    if _WHERE_PATTERN.search(q):
        return "LOCATION"
    if _WHO_PATTERN.search(q):
        if org_word:
            return "ORGANIZATION"
        return "PERSON" if len(core) <= 4 else "PROPER"
    if org_word:
        return "ORGANIZATION"
    if lowered_words & _LOCATION_WORDS:
        return "LOCATION"
    return "PROPER"


def answer_variant(entity_type: str, question: str, answer: str) -> str:
    """The sub-pool key of an answer within its type, "" when the type has none.

    Places and organizations split by the head noun the question asked for, numbers by their form
    (percent, ordinal, bare digits, bare words, or the unit that follows), dates by whether they
    carry a year, years by whether they name a decade, and other proper names by the first content
    word after what or which. The point is that a substitute reads like what it replaces.

    Args:
        entity_type: the answer's ``answer_type``.
        question: the question text.
        answer: the reference answer.
    """
    answer = answer.strip()
    if entity_type in ("LOCATION", "ORGANIZATION"):
        return asking_head(question)
    if entity_type == "NUMBER":
        if "%" in answer:
            return "percent"
        if answer[0] in "$£€":
            return "currency"
        if re.search(r"\d(?:st|nd|rd|th)$", answer):
            return "ordinal"
        unit = (_is_number(answer) or "").lower()
        if unit:
            return unit
        return "digits" if re.search(r"\d", answer) else "words"
    if entity_type == "DATE":
        return "year" if re.search(r"\b\d{3,4}\b", answer) else "noyear"
    if entity_type == "YEAR":
        return "decade" if answer.lower().endswith("s") else "year"
    if entity_type == "PROPER":
        match = _PROPER_HEAD_PATTERN.search(question.lower())
        word = match.group(1) if match else ""
        return "" if word in _VAGUE_HEADS else word
    return ""


def _pool_form(entity_type: str, answer: str) -> str:
    """How an answer is stored for drawing: a bare year, and no leading article on names.

    A stored "in 1932" would turn "In 1987" into "In in 1932", and a stored "the Beatles" would
    turn "of Paris" into "of the the Beatles" whenever the original had no article.
    """
    if entity_type == "YEAR":
        match = _YEAR_PATTERN.match(answer)
        return match.group(1) + ("s" if answer.lower().endswith("s") else "")
    words = answer.split()
    if len(words) > 1 and words[0].lower() == "the":
        return " ".join(words[1:])
    return answer


def _occurs(text: str, needle: str) -> bool:
    if not needle or not text:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", text, re.I) is not None


class Gazetteer:
    """Distinct answers by entity type (and variant), built from (question, answer) pairs."""

    def __init__(self) -> None:
        self._by_type: Dict[str, List[str]] = {t: [] for t in SWAPPABLE}
        self._by_variant: Dict[Tuple[str, str], List[str]] = {}
        self._seen: set = set()

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[str, str]]) -> "Gazetteer":
        """Build from ``(question, answer)`` pairs. Unswappable and empty answers are skipped.

        Args:
            pairs: one pair per question; the answer is the reference whose type is read.
        """
        gaz = cls()
        for question, answer in pairs:
            answer = (answer or "").strip()
            entity_type = answer_type(question, answer)
            if entity_type not in SWAPPABLE:
                continue
            key = (entity_type, normalize(answer))
            if not key[1] or key in gaz._seen:
                continue
            gaz._seen.add(key)
            stored = _pool_form(entity_type, answer)
            gaz._by_type[entity_type].append(stored)
            variant = answer_variant(entity_type, question, answer)
            if variant:
                gaz._by_variant.setdefault((entity_type, variant), []).append(stored)
        return gaz

    def size(self, entity_type: str, variant: str = "") -> int:
        """How many distinct answers a draw of this type (and variant) can pick from.

        Args:
            entity_type: one of ``SWAPPABLE``.
            variant: optional sub-pool key, see ``answer_variant``.
        """
        if variant:
            return len(self._by_variant.get((entity_type, variant), ()))
        return len(self._by_type.get(entity_type, ()))

    def draw(self, entity_type: str, exclude: Sequence[str], rng: random.Random,
             avoid_text: str = "", variant: str = "") -> Optional[str]:
        """A substitute of ``entity_type``, or None when no member passes the checks.

        A member is rejected when it normalizes equal to, contains or is contained in any string of
        ``exclude``, or when it already occurs (case-insensitively, on word boundaries) in
        ``avoid_text``. The variant pool is used when it holds at least ``MIN_VARIANT_POOL``
        members, the whole type pool otherwise, except for numbers and other proper names, where a
        small variant pool means None (a distance swapped for a duration, or a port for a protein,
        reads wrongly enough to measure the wrong thing).

        Args:
            entity_type: one of ``SWAPPABLE``.
            exclude: strings the substitute must not equal, contain or sit inside (every alias of
                the original answer).
            rng: the caller's generator, so a draw is reproducible.
            avoid_text: text the substitute is going into; it must not already appear there.
            variant: optional sub-pool key, see ``answer_variant``.
        """
        pool = self._by_variant.get((entity_type, variant), []) if variant else []
        if len(pool) < MIN_VARIANT_POOL:
            if entity_type in _STRICT_VARIANT_TYPES:
                return None
            pool = self._by_type.get(entity_type, [])
        if not pool:
            return None
        excluded = [e for e in (normalize(x) for x in exclude) if e]

        def acceptable(candidate: str) -> bool:
            norm = normalize(candidate)
            if not norm:
                return False
            if any(norm == e or norm in e or e in norm for e in excluded):
                return False
            return not _occurs(avoid_text, candidate)

        for _ in range(min(len(pool), 64)):
            candidate = pool[rng.randrange(len(pool))]
            if acceptable(candidate):
                return candidate
        order = list(range(len(pool)))
        rng.shuffle(order)
        for i in order:
            if acceptable(pool[i]):
                return pool[i]
        return None


def swap_in_text(text: str, original: str, substitute: str) -> Tuple[str, int]:
    """Replace every case-insensitive, whole-word occurrence of ``original`` and count them.

    "Paris" does not match inside "Parisian". A count of 0 means the text is returned unchanged and
    the original was not swappable here.

    Args:
        text: the passage or chunk.
        original: the string to replace.
        substitute: what to put in its place, verbatim.
    """
    if not original:
        return text, 0
    pattern = re.compile(r"(?<!\w)" + re.escape(original) + r"(?!\w)", re.I)
    return pattern.subn(lambda _: substitute, text)


def swap_all_aliases(text: str, aliases: Sequence[str], substitute: str) -> Tuple[str, int]:
    """Swap every alias, longest first, so no shorter alias is replaced inside a longer one.

    Args:
        text: the passage or chunk.
        aliases: every surface form of the answer.
        substitute: what to put in place of each, verbatim.

    Returns:
        The swapped text and the total number of replacements.
    """
    total = 0
    for alias in sorted({a for a in aliases if a}, key=len, reverse=True):
        text, n = swap_in_text(text, alias, substitute)
        total += n
    return text, total


def alias_remains(text: str, aliases: Sequence[str]) -> bool:
    """True when any alias still occurs in ``text`` as a whole word, case-insensitively.

    Args:
        text: the swapped passage or chunk.
        aliases: every surface form of the answer.
    """
    return any(swap_in_text(text, a, "")[1] for a in aliases if a)


# ------------------------------------------------------------------ counterfactual bookkeeping

FREQ_STRATA = ("0", "1-9", "10-99", "100-999", "1000+")
INELIGIBLE_REASONS = ("unanswerable", "type_not_swappable", "answer_not_in_passage", "no_substitute",
                       "alias_remains")
# fewest items a stratum needs before its gate line is read as anything
MIN_STRATUM_ITEMS = 30
MAX_MEMORIZATION_RATIO = 0.05
MIN_FOLLOW_RATE = 0.90

# per-record fields that are token arrays, evidence packaging or internal state, never JSON
_NON_JSON_FIELDS = frozenset((
    "prompt_ids", "forced_ids", "forced_mask", "evidence_row", "answer_forced", "gold_chunks", "cf",
))


def frequency_stratum(count: int) -> str:
    """Bucket of an answer's corpus count, one of ``FREQ_STRATA``.

    Args:
        count: how often the answer's token sequence occurs in the reference corpus.
    """
    if count <= 0:
        return FREQ_STRATA[0]
    if count < 10:
        return FREQ_STRATA[1]
    if count < 100:
        return FREQ_STRATA[2]
    if count < 1000:
        return FREQ_STRATA[3]
    return FREQ_STRATA[4]


def sigmoid_ratio(ll_original: float, ll_swapped: float) -> float:
    """``sigmoid(ll_original - ll_swapped)``: how much of the preference sits with the original.

    Near 1 the model scores its remembered answer far above the one the evidence gives, near 0 the
    evidence wins, 0.5 is indifference.

    Args:
        ll_original: sequence log-probability of the original answer under the swapped evidence.
        ll_swapped: the same for the substitute.
    """
    delta = ll_original - ll_swapped
    if delta >= 0:
        return 1.0 / (1.0 + math.exp(-delta))
    z = math.exp(delta)
    return z / (1.0 + z)


def prepare_counterfactual(records: List[dict], gazetteer: Gazetteer, rng: random.Random,
                           tokenizer) -> Counter:
    """Decide once per record whether it can take part, and build its swapped evidence.

    A record is eligible when it is answerable, its first reference has a swappable type, that
    reference occurs in the passage and the gazetteer has a substitute that does not occur in the
    passage and is not any alias. Every alias is swapped, and a record where one still occurs
    afterwards is ineligible with reason ``alias_remains``. Sets ``record["cf"]`` to ``{"original", "substitute", "type",
    "chunks": [(text, ids)], "count"}`` or None, plus ``cf_original``, ``cf_substitute``,
    ``cf_type`` or ``cf_reason``. Records that already carry a ``cf`` entry are left alone (an
    injected-fact source builds its own).

    Args:
        records: records with ``question``, ``references``, ``unanswerable`` and ``gold_chunks``
            (a list of ``(text, token ids)``).
        gazetteer: substitutes, usually built over the slice's own answers.
        rng: the caller's generator.
        tokenizer: callable returning ``{"input_ids": ...}``, used to re-tokenize changed chunks.

    Returns:
        How many records were ineligible, by reason, with the eligible count under ``"eligible"``.
    """
    counts: Counter = Counter()
    for record in records:
        if record.get("cf") is not None:
            counts["eligible"] += 1
            continue
        record["cf"] = None
        references = record.get("references") or []
        if record.get("unanswerable") or not references:
            record["cf_reason"] = "unanswerable"
        else:
            original = references[0]
            entity_type = answer_type(record["question"], original)
            passage = "\n".join(text for text, _ in record["gold_chunks"])
            if entity_type not in SWAPPABLE:
                record["cf_reason"] = "type_not_swappable"
            elif swap_in_text(passage, original, "")[1] == 0:
                record["cf_reason"] = "answer_not_in_passage"
            else:
                substitute = gazetteer.draw(
                    entity_type, references, rng, avoid_text=passage,
                    variant=answer_variant(entity_type, record["question"], original),
                )
                if substitute is None:
                    record["cf_reason"] = "no_substitute"
                else:
                    chunks, total = [], 0
                    leftover = False
                    for text, ids in record["gold_chunks"]:
                        swapped, n = swap_all_aliases(text, references, substitute)
                        total += n
                        leftover = leftover or alias_remains(swapped, references)
                        if n:
                            ids = tokenizer(swapped, add_special_tokens=False)["input_ids"]
                        if ids:
                            chunks.append((swapped, ids))
                    if leftover:
                        record["cf_reason"] = "alias_remains"
                    else:
                        record["cf"] = {"original": original, "substitute": substitute,
                                        "type": entity_type, "chunks": chunks, "count": total}
                        record["cf_original"], record["cf_substitute"] = original, substitute
                        record["cf_type"] = entity_type
        if record["cf"] is None:
            counts[record["cf_reason"]] += 1
        else:
            counts["eligible"] += 1
    return counts


def _stratum_order(name: str) -> Tuple[int, float]:
    if name in FREQ_STRATA:
        return 0, FREQ_STRATA.index(name)
    try:
        return 1, float(name)
    except ValueError:
        return 2, 0.0


def _stratum_reading(rows: List[dict]) -> dict:
    n = len(rows)
    n_follow = sum(1 for r in rows if r.get("cf_follow"))
    n_stuck = sum(1 for r in rows if r.get("cf_stuck"))
    mr_ll = [r["mr_ll"] for r in rows if r.get("mr_ll") is not None and not math.isnan(r["mr_ll"])]
    follow_rate = n_follow / n if n else None
    mr_gen = n_stuck / (n_stuck + n_follow) if (n_stuck + n_follow) else None
    if n < MIN_STRATUM_ITEMS:
        gate = "n too small"
    else:
        passed = (mr_gen is not None and mr_gen <= MAX_MEMORIZATION_RATIO
                  and follow_rate >= MIN_FOLLOW_RATE)
        gate = "PASS" if passed else "FAIL"
    return {
        "n": n, "n_follow": n_follow, "n_stuck": n_stuck, "follow_rate": follow_rate,
        "mr_gen": mr_gen, "mr_ll": sum(mr_ll) / len(mr_ll) if mr_ll else None,
        "other_rate": (n - n_follow - n_stuck) / n if n else None, "gate": gate,
    }


def counterfactual_readings(records: List[dict]) -> dict:
    """Per stratum readings of the counterfactual condition, over two item sets.

    ``"correct_with_gold"`` keeps the items answered correctly with the unswapped chunk
    (``gold_em`` 1), which is the published memorization-ratio population; ``"all_eligible"`` keeps
    every eligible item. Each holds ``{stratum: reading}`` plus ``"all"`` pooled over strata, and a
    reading is ``n``, ``follow_rate``, ``mr_gen = n_stuck / (n_stuck + n_follow)`` (None when both
    are 0), the mean ``mr_ll``, ``other_rate`` and a ``gate`` of ``PASS`` (``mr_gen <= 0.05`` and
    ``follow_rate >= 0.90``), ``FAIL`` or ``n too small`` (under 30 items).

    Args:
        records: eligible records after the counterfactual condition ran, carrying ``cf_follow``,
            ``cf_stuck``, ``mr_ll``, ``stratum`` and ``gold_em``.
    """
    out = {}
    for label, keep in (("correct_with_gold", lambda r: r.get("gold_em") == 1.0),
                        ("all_eligible", lambda r: True)):
        rows = [r for r in records if keep(r)]
        by_stratum: Dict[str, List[dict]] = {}
        for r in rows:
            by_stratum.setdefault(str(r.get("stratum", "all")), []).append(r)
        readings = {name: _stratum_reading(items)
                    for name, items in sorted(by_stratum.items(), key=lambda kv: _stratum_order(kv[0]))}
        readings["all"] = _stratum_reading(rows)
        out[label] = readings
    return out


def record_for_json(record: dict) -> dict:
    """A record without its token arrays, evidence packaging and internal state.

    Args:
        record: a per-question evaluation record.
    """
    return {k: v for k, v in record.items() if k not in _NON_JSON_FIELDS}


def counterfactual_evidence_row(chunks: Sequence[Tuple[str, List[int]]],
                                embedder) -> Optional[dict]:
    """The evidence row ``{"ids", "chunk_ids", "keys"}`` for swapped chunks, keys re-embedded.

    Args:
        chunks: ``(text, token ids)`` per chunk.
        embedder: anything with ``.encode(list of str) -> [N, 384]``.
    """
    ids: List[int] = []
    chunk_ids: List[int] = []
    for index, (_, chunk) in enumerate(chunks):
        ids.extend(chunk)
        chunk_ids.extend([index] * len(chunk))
    if not ids:
        return None
    return {"ids": ids, "chunk_ids": chunk_ids, "keys": embedder.encode([t for t, _ in chunks])}
