"""Fictional biographies for fact injection: pools, people, renders, probes.

Every value is invented from a seeded syllable generator, so the model has no prior for any person's
facts. Each attribute of a person is drawn **uniformly and independently** from its pool, but the
corpus frequency of a value is not uniform: a person's tier multiplies how often their value is
written, so a model that learns only the value marginals, with no binding to a name, already ranks
the gold of a heavily exposed person above 0.5. Chance for a closed-book reading is therefore read
against a control, not assumed: ``FreshNames`` makes a name that occurs nowhere in the corpus, and
the same probe with that name measures what the value marginals alone give.

Pure Python (no torch, no tokenizer, no numpy), so a builder, a scorer and a test can all import it.

The module is also the source of exact fact spans. A render returns the character range of every
attribute value and every name mention, so a loss mask built from them is exact by construction and
no tagger is needed.

Args:
    (module) ATTRIBUTES: the five facts of a person, in the order a factspan code names them.
    (module) TIERS: how often a person's biography is rendered into the corpus.
"""
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

ATTRIBUTES = ("birth_date", "birth_city", "university", "major", "employer")
ENTITY_ATTRIBUTES = ("birth_city", "university", "employer")
DATE_ATTRIBUTES = ("birth_date",)
NOUN_ATTRIBUTES = ("major",)
TIERS = (1, 10, 100, 1000)

DEFAULT_PERSONS_PER_TIER = {1000: 100, 100: 500, 10: 1000, 1: 2000}
POOL_SIZES = {"birth_city": 200, "university": 150, "employer": 150, "major": 100, "birth_date": 1000}

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")

_ONSETS = ("b", "br", "d", "dr", "f", "g", "gr", "k", "kr", "l", "m", "n", "p", "pr", "r", "s", "st",
           "t", "th", "v", "z", "sh", "ch", "kh", "j", "w", "h", "tr", "fl", "gl")
_VOWELS = ("a", "e", "i", "o", "u", "ae", "ia", "ou", "ei", "oa")
_CODAS = ("n", "r", "l", "s", "th", "m", "x", "nd", "rk")
_CITY_PREFIXES = ("Port", "Mount", "New", "Saint", "Lake", "Fort", "East", "Glen")
_DISCIPLINES = ("linguistics", "mechanics", "ecology", "chemistry", "engineering", "semantics",
                "dynamics", "architecture", "geometry", "pharmacology", "cartography", "acoustics")


def attribute_class(attribute: str) -> str:
    """``"entity"``, ``"date"`` or ``"noun"``, the class a reading is split by.

    Args:
        attribute: one of ``ATTRIBUTES``.
    """
    if attribute in ENTITY_ATTRIBUTES:
        return "entity"
    if attribute in DATE_ATTRIBUTES:
        return "date"
    if attribute in NOUN_ATTRIBUTES:
        return "noun"
    raise ValueError(f"unknown attribute {attribute!r}")


@dataclass
class Person:
    """One fictional person.

    Attributes:
        person_id: dense integer id, also the row of the person's card in the biography store.
        name: ``"First Middle Last"``, unique across people.
        tier: exposures in the corpus, one of ``TIERS``.
        attributes: a value for every key of ``ATTRIBUTES``.
        store_chunk: canonical store text; holds every attribute value verbatim.
    """
    person_id: int
    name: str
    tier: int
    attributes: Dict[str, str]
    store_chunk: str


class _Registry:
    """Strings already handed out, to keep every new one from containing or sitting inside another.

    That containment rule is what lets a plain substring search stand in for "this value occurs in
    this text" and a plain replace stand in for a swap: no value can be hiding inside another.
    """

    def __init__(self, seen: Sequence[str] = ()):
        self.seen: List[str] = [s.lower() for s in seen]

    def ok(self, text: str) -> bool:
        low = text.lower()
        return not any(low in s or s in low for s in self.seen)

    def add(self, text: str) -> None:
        self.seen.append(text.lower())


def _stem(rng: random.Random, max_len: int = 9) -> str:
    """A pronounceable invented word, lowercase, 5 letters up to ``max_len``."""
    while True:
        syllables = rng.choice((2, 3, 3))
        parts = []
        for k in range(syllables):
            part = rng.choice(_ONSETS) + rng.choice(_VOWELS)
            if k == syllables - 1 and rng.random() < 0.4:
                part += rng.choice(_CODAS)
            parts.append(part)
        word = "".join(parts)
        if 5 <= len(word) <= max_len:
            return word


def _fresh_stem(rng: random.Random, registry: _Registry) -> str:
    while True:
        stem = _stem(rng)
        if registry.ok(stem):
            registry.add(stem)
            return stem


def _whitespace_prefix_free(values: Sequence[str]) -> bool:
    """True when no value's whitespace tokens are a leading run of another value's."""
    split = [tuple(v.split()) for v in values]
    present = set(split)
    for tokens in split:
        for cut in range(1, len(tokens)):
            if tokens[:cut] in present:
                return False
    return True


def make_pools(seed: int) -> Dict[str, List[str]]:
    """Per attribute, a list of distinct invented values.

    ``birth_city`` 200, ``university`` 150, ``employer`` 150, ``major`` 100, ``birth_date`` 1000.
    Values are distinct within a pool, no value is a whitespace-token prefix of another in its pool,
    and no value of any non-date pool contains or sits inside another value.

    Args:
        seed: the pool seed; the same seed gives the same pools.
    """
    rng = random.Random(f"pools:{seed}")
    registry = _Registry()
    pools: Dict[str, List[str]] = {}

    cities = []
    while len(cities) < POOL_SIZES["birth_city"]:
        stem = _stem(rng).capitalize()
        value = f"{rng.choice(_CITY_PREFIXES)} {stem}" if rng.random() < 0.3 else stem
        if registry.ok(value) and registry.ok(stem):
            registry.add(value)
            cities.append(value)
    pools["birth_city"] = cities

    universities = []
    while len(universities) < POOL_SIZES["university"]:
        stem = _stem(rng, 7).capitalize()
        form = rng.choice(("University of {s}", "{s} Institute of Technology", "{s} College"))
        value = form.format(s=stem)
        if registry.ok(value) and registry.ok(stem):
            registry.add(value)
            universities.append(value)
    pools["university"] = universities

    employers = []
    while len(employers) < POOL_SIZES["employer"]:
        form = rng.choice(("{s} Systems", "{s} Holdings", "{s} and {t}"))
        # two stems in one value: short ones keep the value within 8 tokens
        short = 5 if "{t}" in form else 9
        first = _stem(rng, short).capitalize()
        second = _stem(rng, short).capitalize() if "{t}" in form else ""
        value = form.format(s=first, t=second)
        if registry.ok(value) and registry.ok(first) and (not second or registry.ok(second)):
            registry.add(value)
            employers.append(value)
    pools["employer"] = employers

    majors = []
    while len(majors) < POOL_SIZES["major"]:
        stem = _stem(rng)
        value = f"{stem}{rng.choice(('ic', 'al', 'an'))} {rng.choice(_DISCIPLINES)}"
        if registry.ok(value) and registry.ok(stem):
            registry.add(value)
            majors.append(value)
    pools["major"] = majors

    dates = set()
    while len(dates) < POOL_SIZES["birth_date"]:
        dates.add(f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(1950, 2005)}")
    pools["birth_date"] = sorted(dates, key=lambda d: hashlib.sha1(d.encode()).digest())

    for attribute, values in pools.items():
        assert len(set(values)) == len(values), f"duplicate values in the {attribute} pool"
        assert _whitespace_prefix_free(values), f"a {attribute} value prefixes another"
    return pools


def make_people(pools: Dict[str, List[str]], persons_per_tier: Dict[int, int], seed: int) -> List[Person]:
    """People with unique names and uniform independent attributes.

    Ids run from 0 in descending tier order, then draw order, so the same arguments always give the
    same list.

    Args:
        pools: the output of ``make_pools``.
        persons_per_tier: ``{tier: count}``; every tier must be one of ``TIERS``.
        seed: the people seed.
    """
    for tier in persons_per_tier:
        assert tier in TIERS, f"tier {tier} is not one of {TIERS}"
    rng = random.Random(f"people:{seed}")
    registry = _Registry([v for values in pools.values() for v in values])
    firsts = [_fresh_stem(rng, registry).capitalize() for _ in range(300)]
    middles = [_fresh_stem(rng, registry).capitalize() for _ in range(300)]
    lasts = [_fresh_stem(rng, registry).capitalize() for _ in range(400)]

    people: List[Person] = []
    names = set()
    for tier in sorted(persons_per_tier, reverse=True):
        for _ in range(persons_per_tier[tier]):
            while True:
                name = f"{rng.choice(firsts)} {rng.choice(middles)} {rng.choice(lasts)}"
                if name not in names:
                    names.add(name)
                    break
            attributes = {a: rng.choice(pools[a]) for a in ATTRIBUTES}
            person = Person(len(people), name, tier, attributes, "")
            person.store_chunk = render_store_chunk(person)
            people.append(person)
    return people


class FreshNames:
    """Deterministic names that occur nowhere in a facts file: the prior control of a closed-book probe.

    A fresh name has the shape of a real one (first, middle and last stem) but every word is absent
    from every person's name and does not sit inside any pool value, so a model has no binding to
    read for it and can only answer from the marginal frequency of the values.

    Args:
        people: every person of the facts file.
        pools: the pools the people were drawn from.
    """

    def __init__(self, people: Sequence[Person], pools: Dict[str, List[str]]):
        self.names = {p.name for p in people}
        self.words = {w.lower() for name in self.names for w in name.split()}
        self.pool_text = "\n".join(v.lower() for values in pools.values() for v in values)

    def name(self, key: str) -> str:
        """The fresh name for ``key``; the same key always gives the same name.

        Args:
            key: any string, normally the item id.
        """
        digest = hashlib.sha1(f"fresh_name:{key}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        words: List[str] = []
        while len(words) < 3:
            stem = _stem(rng)
            if stem not in self.words and stem not in self.pool_text and stem not in words:
                words.append(stem)
        name = " ".join(w.capitalize() for w in words)
        assert name not in self.names and not any(w in self.words for w in words), name
        return name


# {s} is the subject (the full name or a pronoun), {v} the attribute value. template 0 is the
# in-distribution probe form: the value is sentence final. no template starts with a lowercase
# value, so capitalizing a sentence start never changes a value
_TEMPLATES = {
    "birth_city": (
        "{s} was born in {v}.",
        "The city of {v} is where {s} was born.",
        "{v} is the hometown of {s}.",
        "Born in {v}, {s} began life in a small household.",
        "The birthplace of {s} is {v}.",
        "{s} first saw the light of day in {v}.",
    ),
    "birth_date": (
        "{s} was born on {v}.",
        "{s} entered the world on {v}.",
        "The date of birth for {s} is {v}.",
        "On {v}, {s} was born.",
        "{v} marks the day {s} was born.",
        "{s} has a birthday that falls on {v}.",
    ),
    "university": (
        "{s} studied at {v}.",
        "{s} graduated from {v}.",
        "{v} is where {s} completed a degree.",
        "{s} attended {v} as a student.",
        "After school, {s} enrolled at {v}.",
        "{v} awarded a degree to {s}.",
    ),
    "major": (
        "{s} majored in {v}.",
        "{s} focused on {v} during college.",
        "At university, {s} specialized in {v}.",
        "{s} earned a degree in {v}.",
        "The field {s} concentrated on was {v}.",
        "{s} chose {v} as an area of study.",
    ),
    "employer": (
        "{s} works for {v}.",
        "{s} is employed by {v}.",
        "{s} has a job at {v}.",
        "{v} employs {s}.",
        "These days, {s} draws a salary from {v}.",
        "The employer of {s} is {v}.",
    ),
}

_PROBES = {
    "birth_city": ("{name} was born in", "Q: Where was {name} born?\nA:"),
    "birth_date": ("{name} was born on", "Q: When was {name} born?\nA:"),
    "university": ("{name} studied at", "Q: Which university did {name} attend?\nA:"),
    "major": ("{name} majored in", "Q: What did {name} study?\nA:"),
    "employer": ("{name} works for", "Q: Who employs {name}?\nA:"),
}
_QUESTIONS = {
    "birth_city": "Where was {name} born?",
    "birth_date": "When was {name} born?",
    "university": "Which university did {name} attend?",
    "major": "What did {name} study?",
    "employer": "Who employs {name}?",
}

_PLACEHOLDER = re.compile(r"(\{s\}|\{v\})")


def _pronoun(person: Person) -> str:
    """A subject pronoun fixed by the person's name alone, so no field has to store it."""
    return "he" if hashlib.sha1(person.name.encode()).digest()[0] & 1 else "she"


def _build_sentence(template: str, subject: str, subject_is_name: bool, value: str,
                    attribute: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    text, spans = "", []
    for part in _PLACEHOLDER.split(template):
        if part == "{s}":
            shown = subject if subject_is_name or text else subject.capitalize()
            if subject_is_name:
                spans.append((len(text), len(text) + len(shown), "name"))
            text += shown
        elif part == "{v}":
            spans.append((len(text), len(text) + len(value), attribute))
            text += value
        else:
            text += part
    return text, spans


def render_bio(person: Person, rng: random.Random, *, name_override: Optional[str] = None,
               value_overrides: Optional[Dict[str, str]] = None
               ) -> Tuple[str, List[Tuple[int, int, str]]]:
    """One biography document: five sentences in a random order, each from a random template.

    The first sentence names the person in full; each later one does with probability 0.5 and
    otherwise uses a pronoun. The draws never depend on the overrides, so a render with overrides and
    the same ``rng`` state has the same structure as the plain render and differs only in the
    substituted text.

    Args:
        person: whose biography to write.
        rng: consumed for the sentence order, the template and the subject of each sentence.
        name_override: written in place of the person's name (a typed placeholder).
        value_overrides: ``{attribute: value}`` written in place of the person's value.

    Returns:
        ``(text, spans)`` with spans ``(char_start, char_end, label)`` for every attribute value and
        every name mention; the label is an attribute name or ``"name"``.
    """
    values = {**person.attributes, **(value_overrides or {})}
    name = name_override if name_override is not None else person.name
    pronoun = _pronoun(person)
    order = list(ATTRIBUTES)
    rng.shuffle(order)
    sentences, spans, cursor = [], [], 0
    for position, attribute in enumerate(order):
        template = rng.choice(_TEMPLATES[attribute])
        use_name = position == 0 or rng.random() < 0.5
        subject = name if use_name else pronoun
        sentence, local = _build_sentence(template, subject, use_name, values[attribute], attribute)
        spans.extend((a + cursor, b + cursor, label) for a, b, label in local)
        sentences.append(sentence)
        cursor += len(sentence) + 1
    return " ".join(sentences), sorted(spans)


def render_store_chunk(person: Person, value_overrides: Optional[Dict[str, str]] = None) -> str:
    """The person's store card, in a form no biography template uses.

    Copying a value from the card into a sentence therefore takes reading, not recall of a string
    seen in training.

    Args:
        person: whose card to write.
        value_overrides: ``{attribute: value}`` written in place of the person's value.
    """
    v = {**person.attributes, **(value_overrides or {})}
    return (f"{person.name}. Born {v['birth_date']} in {v['birth_city']}. "
            f"Education: {v['major']}, {v['university']}. Employer: {v['employer']}.")


def probe_prompt(attribute: str, name: str, form: str) -> Tuple[str, str]:
    """``(context, terminator)`` for a closed-book probe of one attribute.

    Args:
        attribute: one of ``ATTRIBUTES``.
        name: the person's name.
        form: ``"indist"`` (a corpus phrasing, terminator ``"."``) or ``"heldout"`` (a question, never
            in the corpus, terminator ``"\\n"``).
    """
    indist, heldout = _PROBES[attribute]
    if form == "indist":
        return indist.format(name=name), "."
    if form == "heldout":
        return heldout.format(name=name), "\n"
    raise ValueError(f"unknown probe form {form!r}")


def question_for(attribute: str, name: str) -> str:
    """The held-out question text without the ``Q:`` and ``A:`` frame.

    Args:
        attribute: one of ``ATTRIBUTES``.
        name: the person's name.
    """
    return _QUESTIONS[attribute].format(name=name)


def save_facts(path: str, people: List[Person]) -> None:
    """One JSON line per person.

    Args:
        path: output file.
        people: the people to write.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in people:
            f.write(json.dumps({"person_id": p.person_id, "name": p.name, "tier": p.tier,
                                "attributes": p.attributes, "store_chunk": p.store_chunk}) + "\n")


def load_facts(path: str) -> List[Person]:
    """Read a facts file back.

    Args:
        path: a file written by ``save_facts``.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [Person(r["person_id"], r["name"], r["tier"], r["attributes"], r["store_chunk"])
            for r in rows]


def save_pools(path: str, pools: Dict[str, List[str]]) -> None:
    """Write ``{attribute: [values]}``.

    Args:
        path: output file.
        pools: the output of ``make_pools``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pools, f)


def load_pools(path: str) -> Dict[str, List[str]]:
    """Read a pools file back.

    Args:
        path: a file written by ``save_pools``.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
