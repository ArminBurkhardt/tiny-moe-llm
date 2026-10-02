"""Synthetic multi-hop chain questions over fictional entities, for measuring composition.

A question names one start entity and asks for the end of a chain of typed relations, for example
"Which city is the founder of the company that employs Kiva Dorn from?". The facts that answer it
arrive as a buffer of one-sentence chunks (the gold chain plus distractors), so the only way to the
answer is to read the first fact, then use what it returned to find the second, and so on. Every
entity is drawn fresh per question from large generated pools, so no fact is consistent across
questions and nothing can be memorized from the corpus: a model that answers is composing reads.

The distractors are built so a shortcut has nothing to hold on to:

    A   same relation as a gold hop, other entities (supplies the type matched wrong answers)
    B   a wrong bridge: two chained facts off the true chain, so following a wrong first hop leads
        to a wrong answer (for one hop there is no bridge, so ``B1``: other relations of the start
        entity)
    C   one complete decoy chain of the same relations from a start entity that shares a name part
        with the true start

``shortcut_baselines`` measures five leak detectors on a list of questions, and the test bounds
them: a generator whose answer can be picked by frequency or by the question's own subject would
make every composition reading meaningless.

The module is pure Python and builds no embedder and no tokenizer. The statistics helpers at the
bottom (``cell_stats``, ``validity_report``) live here rather than in the eval script so they can be
tested without a GPU.
"""
import math
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

PERSON, COMPANY, CITY, COUNTRY, UNIVERSITY = "PERSON", "COMPANY", "CITY", "COUNTRY", "UNIVERSITY"
ENTITY_TYPES = (PERSON, COMPANY, CITY, COUNTRY, UNIVERSITY)

POOL_SEED = 20240601
MIN_CANDIDATES = 4
DISTRACTOR_RANGE = (6, 14)
HOP_WEIGHTS = (0.25, 0.5, 0.25)

CHUNK_KINDS = ("gold", "A", "B", "B1", "C")


@dataclass(frozen=True)
class Relation:
    """One typed, single-valued relation.

    Attributes:
        name: key in ``RELATIONS``.
        domain: entity type of the subject.
        range: entity type of the object.
        templates: fact sentences with ``{s}`` and ``{o}``; a chunk draws one at random so the
            sentence shape never says whether a chunk is gold.
        phrase: noun phrase for the object given an expression ``{x}`` for the subject.
        question: wh-frame asking for the object given an expression ``{x}`` for the subject.
    """

    name: str
    domain: str
    range: str
    templates: Tuple[str, ...]
    phrase: str
    question: str


RELATIONS: Dict[str, Relation] = {r.name: r for r in (
    Relation("employer", PERSON, COMPANY,
             ("{s} works for {o}.", "{o} employs {s}.", "{s} is employed by {o}.",
              "{s} has a job at {o}."),
             "the company that employs {x}", "Which company employs {x}?"),
    Relation("mentor", PERSON, PERSON,
             ("{s} was mentored by {o}.", "{o} mentored {s}.", "{s} learned the trade under {o}."),
             "the mentor of {x}", "Who mentored {x}?"),
    Relation("birthplace", PERSON, CITY,
             ("{s} was born in {o}.", "{o} is the birthplace of {s}.", "{s} is a native of {o}."),
             "the city where {x} was born", "In which city was {x} born?"),
    Relation("founder", COMPANY, PERSON,
             ("{s} was founded by {o}.", "{o} founded {s}.", "{o} is the founder of {s}."),
             "the founder of {x}", "Who founded {x}?"),
    Relation("headquarters", COMPANY, CITY,
             ("{s} is headquartered in {o}.", "{o} is home to the headquarters of {s}.",
              "{s} has its main offices in {o}."),
             "the city where {x} is headquartered", "In which city is {x} headquartered?"),
    Relation("alma_mater", PERSON, UNIVERSITY,
             ("{s} graduated from {o}.", "{o} counts {s} among its graduates.",
              "{s} earned a degree at {o}."),
             "the university {x} graduated from", "Which university did {x} graduate from?"),
    Relation("campus", UNIVERSITY, CITY,
             ("{s} is located in {o}.", "{o} is where {s} has its campus.",
              "The campus of {s} stands in {o}."),
             "the city where {x} is located", "In which city is {x} located?"),
    Relation("country", CITY, COUNTRY,
             ("{s} is a city in {o}.", "{o} is the country where {s} lies.", "{s} lies in {o}."),
             "the country {x} is in", "In which country is {x}?"),
)}


# ----------------------------------------------------------------------------- composition tuples

def compositions(length: int) -> List[Tuple[str, ...]]:
    """Every valid relation tuple of this length, in a fixed order.

    A tuple is valid when each relation's domain is the previous relation's range.

    Args:
        length: number of hops.
    """
    out: List[Tuple[str, ...]] = [(name,) for name in RELATIONS]
    for _ in range(length - 1):
        out = [t + (n,) for t in out for n, r in RELATIONS.items()
               if r.domain == RELATIONS[t[-1]].range]
    return out


def _bigrams(composition: Sequence[str]) -> List[Tuple[str, str]]:
    return [(composition[i], composition[i + 1]) for i in range(len(composition) - 1)]


def split_compositions(seed: int, held_out_fraction: float = 0.2) -> Tuple[Set[Tuple[str, ...]], Set[Tuple[str, ...]]]:
    """Split the length 2 tuples into seen and held out.

    The held out ones never appear as 2-hop questions in training. The draw is repeated (seeded)
    until every relation that occurs in a 2-hop tuple still occurs in a seen one, so the held out
    test is about the pairing and not about a relation the model never saw composed.

    Args:
        seed: draw seed.
        held_out_fraction: share of the length 2 tuples held out, at least 2 tuples.

    Returns:
        ``(seen, held_out)`` as sets of 2-tuples.
    """
    allowed = compositions(2)
    n_held = max(2, round(held_out_fraction * len(allowed)))
    rng = random.Random(f"split_compositions:{seed}")
    relations = {r for t in allowed for r in t}
    for _ in range(1000):
        shuffled = allowed[:]
        rng.shuffle(shuffled)
        held, seen = set(shuffled[:n_held]), set(shuffled[n_held:])
        if {r for t in seen for r in t} == relations:
            return seen, held
    raise RuntimeError("no held out split keeps every relation in a seen composition")


def seen_compositions(length: int, held_out: Set[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    """Tuples a training or seen-composition eval question of this length may use.

    Length 3 drops every tuple containing a held out pair: a 3-hop question holding a held out
    pair would teach that exact pairing and void the held out test. Length 1 and 4 are unrestricted
    (length 4 only ever appears in its own split).

    Args:
        length: number of hops.
        held_out: the held out 2-tuples from ``split_compositions``.
    """
    all_tuples = compositions(length)
    if length < 3:
        return all_tuples if length == 1 else [t for t in all_tuples if t not in held_out]
    return [t for t in all_tuples if not any(b in held_out for b in _bigrams(t))]


def contains_held_out(composition: Sequence[str], held_out: Set[Tuple[str, ...]]) -> bool:
    """Whether any consecutive pair of the composition is a held out pair."""
    return any(b in held_out for b in _bigrams(composition))


# --------------------------------------------------------------------------------- entity pools

_ONSETS = ["b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr",
           "kr", "tr", "vl", "sh", "th", "st", "gl", "pr", "sk", "fl", "cr"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "io", "ou", "ia"]
_CODAS = ["", "", "n", "r", "l", "s", "m", "k", "th", "nd", "rn"]

COMPANY_FORMS = ("{s} Systems", "{s} Holdings", "{s} Industries", "{s} Labs", "{s} Dynamics",
                 "{s} Partners", "{s} Logistics", "{s} Works")
CITY_FORMS = ("Port {s}", "{s} Falls", "{s} Heights", "{s} Harbor", "Fort {s}", "{s} Springs",
              "{s} Ridge", "{s} Crossing")
UNIVERSITY_FORMS = ("University of {s}", "{s} College", "{s} Institute of Technology",
                    "{s} Polytechnic", "{s} Academy")
COUNTRY_FORMS = ("{s}", "{s}land", "{s}stan")
_FORMS = {COMPANY: COMPANY_FORMS, CITY: CITY_FORMS, UNIVERSITY: UNIVERSITY_FORMS,
          COUNTRY: COUNTRY_FORMS}


def _names(rng: random.Random, n: int, syllables: Tuple[int, int], min_len: int = 5) -> List[str]:
    seen, out = set(), []
    while len(out) < n:
        k = rng.randint(*syllables)
        word = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS) for _ in range(k))
        word = word.capitalize()
        if len(word) >= min_len and word not in seen:
            seen.add(word)
            out.append(word)
    return out


@lru_cache(maxsize=1)
def pools() -> Dict[str, List[str]]:
    """The fixed fictional name pools: first and last names and one stem list per place type."""
    rng = random.Random(POOL_SEED)
    return {
        "first": _names(rng, 3000, (2, 3)),
        "last": _names(rng, 3000, (2, 3)),
        COMPANY: _names(rng, 2500, (2, 3)),
        CITY: _names(rng, 2500, (2, 3)),
        UNIVERSITY: _names(rng, 1500, (2, 3)),
        COUNTRY: _names(rng, 300, (2, 3)),
    }


def _stem_of(etype: str, text: str) -> Optional[str]:
    for form in _FORMS.get(etype, ()):
        head, tail = form.split("{s}")
        if text.startswith(head) and text.endswith(tail) and len(text) > len(head) + len(tail):
            return text[len(head):len(text) - len(tail)]
    return None


class _World:
    """Entities already used in one question, so every new one is distinct and substring free."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: List[str] = []
        self.pools = pools()

    def free(self, text: str) -> bool:
        low = text.lower()
        return not any(low in u or u in low for u in self.used)

    def _take(self, text: str) -> Optional[str]:
        if self.free(text):
            self.used.append(text.lower())
            return text
        return None

    def draw(self, etype: str) -> str:
        for _ in range(500):
            if etype == PERSON:
                text = f"{self.rng.choice(self.pools['first'])} {self.rng.choice(self.pools['last'])}"
            else:
                text = self.rng.choice(_FORMS[etype]).format(s=self.rng.choice(self.pools[etype]))
            got = self._take(text)
            if got:
                return got
        raise RuntimeError(f"could not draw a free {etype}")

    def variant(self, etype: str, text: str) -> str:
        """A new entity sharing one name part with ``text`` (first or last name, or the stem)."""
        for _ in range(500):
            if etype == PERSON:
                first, last = text.split(" ")
                if self.rng.random() < 0.5:
                    cand = f"{first} {self.rng.choice(self.pools['last'])}"
                else:
                    cand = f"{self.rng.choice(self.pools['first'])} {last}"
            else:
                stem = _stem_of(etype, text)
                cand = self.rng.choice(_FORMS[etype]).format(s=stem)
            got = self._take(cand) if cand != text else None
            if got:
                return got
        raise RuntimeError(f"could not draw a variant of {text}")


# ---------------------------------------------------------------------------------- the question

@dataclass
class ChainQuestion:
    """One question with its buffer.

    Attributes:
        question: the wh question, naming only the start entity.
        answer: the end of the chain.
        answer_type: entity type of the answer.
        hops: number of relations in the chain.
        composition: the relation names, hop 1 first.
        chunks: one sentence per chunk, in buffer order.
        chunk_hop: 0 for a distractor, h for the gold chunk at hop h.
        chunk_kind: ``gold``, ``A``, ``B``, ``B1`` or ``C`` per chunk.
        candidates: the distinct objects of the final relation's chunks (gold, final relation
            distractors, the decoy's last chunk): the answers a reader can pick from the wh-frame
            alone, so ``1 / len(candidates)`` is the chance line.
        entities: the true chain e0..ek.
        chunk_facts: ``(subject, relation, object)`` per chunk, for the leak detectors.
    """

    question: str
    answer: str
    answer_type: str
    hops: int
    composition: Tuple[str, ...]
    chunks: List[str]
    chunk_hop: List[int]
    chunk_kind: List[str]
    candidates: List[str]
    entities: List[str]
    chunk_facts: List[Tuple[str, str, str]] = field(default_factory=list)


def _fact(rng: random.Random, relation: str, subject: str, obj: str) -> str:
    return rng.choice(RELATIONS[relation].templates).format(s=subject, o=obj)


def _question_text(composition: Sequence[str], start: str) -> str:
    expr = start
    for name in composition[:-1]:
        expr = RELATIONS[name].phrase.format(x=expr)
    return RELATIONS[composition[-1]].question.format(x=expr)


def final_relation_objects(facts: Sequence[Tuple[str, str, str]], relation: str) -> List[str]:
    """Distinct objects of the chunks whose relation is ``relation``, in buffer order.

    The question's wh-frame names the final relation, so these are the only entities a reader can
    pick without composing: gold, the final relation distractors and the decoy chain's end.

    Args:
        facts: ``(subject, relation, object)`` per chunk.
        relation: the final relation of the composition.
    """
    out: List[str] = []
    for _, r, o in facts:
        if r == relation and o not in out:
            out.append(o)
    return out


def _build(rng: random.Random, composition: Tuple[str, ...], n_distractors: int) -> ChainQuestion:
    k = len(composition)
    world = _World(rng)
    rels = [RELATIONS[n] for n in composition]
    chain = [world.draw(rels[0].domain)]
    for rel in rels:
        chain.append(world.draw(rel.range))

    facts: List[Tuple[str, str, str]] = []
    kinds: List[str] = []
    hops: List[int] = []

    def add(subject, relation, obj, kind, hop=0):
        facts.append((subject, relation, obj))
        kinds.append(kind)
        hops.append(hop)

    for i, name in enumerate(composition):
        add(chain[i], name, chain[i + 1], "gold", i + 1)

    decoy = [world.variant(rels[0].domain, chain[0])]
    for rel in rels:
        decoy.append(world.draw(rel.range))
    for i, name in enumerate(composition):
        add(decoy[i], name, decoy[i + 1], "C")

    final_domain = rels[-1].domain
    feeders = [r for r in RELATIONS.values() if r.range == final_domain and r.name != composition[-1]]

    def add_final_a():
        # for chains, the gold, decoy and bridge subjects of the final relation all recur in the
        # buffer, so a final relation distractor gets a feeder fact naming its subject too
        subject = world.draw(final_domain)
        add(subject, composition[-1], world.draw(rels[-1].range), "A")
        if k < 2:
            return 1
        feeder = rng.choice(feeders)
        add(world.draw(feeder.domain), feeder.name, subject, "A")
        return 2

    remaining = n_distractors - k
    for _ in range(2):
        remaining -= add_final_a()
    b1_pool = [r for r in RELATIONS.values() if r.domain == rels[0].domain and r.name != composition[0]]
    rng.shuffle(b1_pool)
    while remaining > 0:
        options = ["A"]
        if k >= 2 and remaining >= 2:
            options.append("B")
        if k == 1 and b1_pool:
            options.append("B1")
        pick = rng.choice(options)
        if pick == "A":
            name = rng.choice(composition)
            if name == composition[-1] and remaining >= (1 if k < 2 else 2):
                remaining -= add_final_a()
            else:
                if name == composition[-1]:
                    others = [n for n in composition if n != composition[-1]]
                    name = rng.choice(others) if others else rng.choice(feeders).name
                add(world.draw(RELATIONS[name].domain), name, world.draw(RELATIONS[name].range), "A")
                remaining -= 1
        elif pick == "B":
            i = rng.randrange(k - 1)
            x = world.draw(rels[i].domain)
            mid = world.draw(rels[i].range)
            add(x, composition[i], mid, "B")
            add(mid, composition[i + 1], world.draw(rels[i + 1].range), "B")
            remaining -= 2
        else:
            rel = b1_pool.pop()
            add(chain[0], rel.name, world.draw(rel.range), "B1")
            remaining -= 1

    order = list(range(len(facts)))
    rng.shuffle(order)
    facts = [facts[i] for i in order]
    kinds = [kinds[i] for i in order]
    hops = [hops[i] for i in order]
    chunks = [_fact(rng, r, s, o) for s, r, o in facts]

    answer_type = rels[-1].range
    candidates = final_relation_objects(facts, composition[-1])
    return ChainQuestion(
        question=_question_text(composition, chain[0]), answer=chain[-1], answer_type=answer_type,
        hops=k, composition=composition, chunks=chunks, chunk_hop=hops, chunk_kind=kinds,
        candidates=candidates, entities=chain, chunk_facts=facts,
    )


def _valid(q: ChainQuestion) -> bool:
    low = q.question.lower()
    if q.entities[0].lower() not in low:
        return False
    if any(e.lower() in low for e in q.entities[1:]):
        return False
    final = [c for c, h in zip(q.chunks, q.chunk_hop) if h == q.hops][0]
    if q.answer not in final:
        return False
    holders = [c for c in q.chunks if q.answer in c]
    return len(holders) == 1 and len(q.candidates) >= MIN_CANDIDATES


def min_distractors(hops: int) -> int:
    """Fewest distractor chunks a question of this length can hold.

    The decoy chain takes ``hops``; two final relation distractors take one chunk each for one hop
    and two (the distractor and the feeder naming its subject) for longer chains, which keeps four
    candidate answers.

    Args:
        hops: chain length.
    """
    return hops + (2 if hops < 2 else 4)


def make_question(rng: random.Random, hops: int, composition: Tuple[str, ...],
                  n_distractors: int) -> ChainQuestion:
    """Draw one question and its buffer.

    A draw whose entities collide with the question text or the answer's own chunk is thrown away
    and redrawn, so the asserted properties hold by construction.

    Args:
        rng: the only source of randomness.
        hops: chain length.
        composition: relation names, ``len(composition) == hops``, each relation's domain the
            previous one's range.
        n_distractors: number of distractor chunks, at least ``min_distractors(hops)``.
    """
    assert len(composition) == hops, f"composition {composition} is not {hops} hops"
    for prev, name in zip(composition, composition[1:]):
        assert RELATIONS[prev].range == RELATIONS[name].domain, f"{prev} then {name} is not typed"
    assert n_distractors >= min_distractors(hops), f"{n_distractors} distractors cannot hold a {hops}-hop decoy"
    for _ in range(50):
        q = _build(rng, composition, n_distractors)
        if _valid(q):
            return q
    raise RuntimeError(f"no valid draw for {composition}")


def sample_hops(rng: random.Random, weights: Sequence[float] = HOP_WEIGHTS) -> int:
    """Hops 1, 2, 3 at the given weights (25 / 50 / 25 by default)."""
    return 1 + rng.choices(range(len(weights)), weights=weights)[0]


def sample_question(rng: random.Random, held_out: Set[Tuple[str, ...]], hops: Optional[int] = None,
                    composition: Optional[Tuple[str, ...]] = None) -> ChainQuestion:
    """A question over seen compositions, with the distractor count drawn in 6..14.

    Args:
        rng: the only source of randomness.
        held_out: held out 2-tuples from ``split_compositions``.
        hops: chain length, drawn at 25 / 50 / 25 over 1..3 when None.
        composition: fixes the tuple (used for the held out and 4-hop splits); drawn from the
            seen tuples of this length when None.
    """
    if composition is not None:
        hops = len(composition)
    elif hops is None:
        hops = sample_hops(rng)
    if composition is None:
        composition = rng.choice(seen_compositions(hops, held_out))
    return make_question(rng, hops, composition,
                         max(rng.randint(*DISTRACTOR_RANGE), min_distractors(hops)))


# ----------------------------------------------------------------------------- leak detectors

def _types_of(q: ChainQuestion) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for s, r, o in q.chunk_facts:
        types[s] = RELATIONS[r].domain
        types[o] = RELATIONS[r].range
    return types


def _pick_value(counts: Dict[str, int], answer: str, largest: bool) -> float:
    target = max(counts.values()) if largest else min(counts.values())
    tied = [c for c, n in counts.items() if n == target]
    return 1.0 / len(tied) if answer in tied else 0.0


def shortcut_baselines(questions: List[ChainQuestion]) -> Dict[str, float]:
    """Expected accuracy of answer pickers that never compose, averaged over the questions.

    ``frequent`` picks the most frequent candidate in the buffer, ``infrequent`` the least
    frequent, ``named_subject`` the object of a chunk whose subject the question names (when it is
    a candidate), ``final_object`` a uniform pick among the objects of final relation chunks,
    ``recurring_subject`` a uniform pick among final relation objects whose subject also occurs in
    another chunk (gold and decoy bridges do), and ``random`` is the chance line, a uniform pick
    among the candidates. Ties are scored by their expected value. For two or more hops every
    picker must sit near ``random``.

    Args:
        questions: the questions to score.
    """
    totals = {"frequent": 0.0, "infrequent": 0.0, "named_subject": 0.0, "final_object": 0.0,
              "recurring_subject": 0.0, "random": 0.0}
    for q in questions:
        final = q.composition[-1]
        pool = [o for s, r, o in q.chunk_facts if r == final]
        totals["final_object"] += (1.0 / len(pool)) if q.answer in pool else 0.0
        occurrences: Dict[str, int] = {}
        for s, _, o in q.chunk_facts:
            occurrences[s] = occurrences.get(s, 0) + 1
            occurrences[o] = occurrences.get(o, 0) + 1
        recurring = [o for s, r, o in q.chunk_facts if r == final and occurrences[s] > 1]
        if recurring:
            totals["recurring_subject"] += (1.0 / len(recurring)) if q.answer in recurring else 0.0
        else:
            totals["recurring_subject"] += 1.0 / len(q.candidates)
        counts = {c: 0 for c in q.candidates}
        for s, _, o in q.chunk_facts:
            for text in (s, o):
                if text in counts:
                    counts[text] += 1
        totals["frequent"] += _pick_value(counts, q.answer, True)
        totals["infrequent"] += _pick_value(counts, q.answer, False)
        totals["random"] += 1.0 / len(q.candidates)
        low = q.question.lower()
        named = [o for s, _, o in q.chunk_facts if s.lower() in low and o in counts]
        totals["named_subject"] += (float(named[0] == q.answer) if named else 1.0 / len(q.candidates))
    n = max(len(questions), 1)
    return {k: v / n for k, v in totals.items()}


# --------------------------------------------------------------------------------- eval statistics

def cell_stats(correct: Sequence[bool], chances: Sequence[float]) -> dict:
    """One grid cell: accuracy, its binomial sigma, chance and the z against chance.

    ``z_vs_chance`` divides by the null sigma ``sqrt(chance (1 - chance) / n)`` so a cell at 0 or 1
    accuracy still has a usable denominator.

    Args:
        correct: per question, whether the prediction was the answer.
        chances: per question, ``1 / len(candidates)``.
    """
    n = len(correct)
    if n == 0:
        return {"n": 0, "acc": float("nan"), "sigma": float("nan"), "chance": float("nan"),
                "null_sigma": float("nan"), "z_vs_chance": float("nan")}
    acc = sum(1 for c in correct if c) / n
    chance = sum(chances) / n
    null_sigma = math.sqrt(chance * (1.0 - chance) / n)
    return {"n": n, "acc": acc, "sigma": math.sqrt(acc * (1.0 - acc) / n), "chance": chance,
            "null_sigma": null_sigma, "z_vs_chance": (acc - chance) / null_sigma if null_sigma > 0 else float("nan")}


def validity_report(cells: Sequence[dict], readable_z: float = 3.0) -> dict:
    """The verdict on the instrument from a flat list of grid cells.

    Each cell is ``{"split", "depth", "hops", "kept", **cell_stats}``. A cell with fewer kept read
    sites than hops must sit at chance (within three null sigma). The 1-hop curve at one or more
    kept sites must be at least three null sigma above chance, otherwise the instrument is "not
    readable": it passes nothing, because a floor effect cannot tell a valid cut from a model that
    never learned the task.

    Args:
        cells: grid cells over every split, depth, hop count and kept-site count.
        readable_z: z the 1-hop curve must reach.

    Returns:
        ``{"valid": bool, "readable": bool, "failing": [cell...], "unreadable": [cell...]}``.
    """
    failing, unreadable, one_hop = [], [], 0
    for cell in cells:
        if cell["n"] == 0:
            continue
        if cell["kept"] < cell["hops"]:
            if abs(cell["acc"] - cell["chance"]) > 3.0 * cell["null_sigma"]:
                failing.append(cell)
        if cell["hops"] == 1 and cell["kept"] >= 1:
            one_hop += 1
            if not cell["z_vs_chance"] >= readable_z:
                unreadable.append(cell)
    readable = one_hop > 0 and not unreadable
    return {"valid": (not failing) and readable, "readable": readable, "failing": failing,
            "unreadable": unreadable}
