"""Fixed abstention and hedge phrasings (PLAN.md Steps 12 and 13).

The target of this whole post-training track is *calibrated abstention*, not chain-of-thought
(PLAN.md's "Post-training" section). Two places need to put words in the model's mouth for that:

  * Step 12 -- SQuAD v2's unanswerable split is the primary abstention supervision, and its
    reference answer is literally the empty string, so a phrasing has to be supplied.
  * Step 13 -- the self-labelled calibration set rewrites low-pass-rate targets as abstentions and
    mid-pass-rate targets as hedges.

**Closed, deliberately -- but no longer tiny.** PLAN.md Step 13 asked for "a hedge, from a small
fixed set of phrasings (not free text)", and the acceptance metric (abstention precision/recall,
then ECE of the abstention signal) needs abstentions to be *detectable*, which a closed set makes
trivially reliable. What the real SFT run showed is that "small" was the wrong half of that to
optimize: 7,786 of 11,873 completions came out as the literal string ``"The passage doesn't say."``
Five phrasings over a third of the QA subset makes one ~6-token sentence the cheapest loss
reduction available under per-token CE, and the model took it (docs/CONCLUSION.md, and Phase 2 of
docs/plans/NEXT.md).

So there are now two sets, and the distinction is load-bearing:

  * ``ABSTENTIONS_PASSAGE`` -- the original five. Still what the *eval* forces as a reference
    target (``scripts/eval_abstention.py``'s teacher-forced pass), so that number stays comparable
    across every checkpoint measured so far.
  * ``ABSTENTIONS_PASSAGE_TRAIN`` -- the superset the corpus builder draws from. Spreading the
    supervision over 15 phrasings is Phase 2's fix #4: no single memorized string is cheap any more.

**``is_abstention`` matches the union**, which is not optional. Detection that knew only the
original five would miss an abstention worded with one of the new phrasings, and would therefore
report a *lower* false-abstention rate than the model earns -- flattering exactly the number
Gate P2 exists to check.

Sampling is the caller's job and must be seeded, so a rebuild of the corpus reproduces itself.
"""
from typing import Sequence

# used where a passage/context is present and the question is not answerable from it (SQuAD v2).
# every phrasing names the passage, because that is the actual claim being supervised: "not in the
# provided text", which is weaker and more honest than "I don't know".
ABSTENTIONS_PASSAGE: Sequence[str] = (
    "The passage doesn't say.",
    "That isn't answerable from the passage.",
    "I can't answer that from this passage -- it doesn't contain that information.",
    "The passage doesn't provide that information.",
    "There's no answer to that in the passage.",
)

# Phase 2's added phrasings. Kept as a separate tuple rather than appended to the one above so the
# eval's forced targets (which draw from ABSTENTIONS_PASSAGE) are unchanged and its teacher-forced
# CE stays comparable to the pre-repair numbers.
ABSTENTIONS_PASSAGE_EXTRA: Sequence[str] = (
    "The passage doesn't mention that.",
    "That's not stated in the passage.",
    "The passage doesn't cover that.",
    "I don't see that in the passage.",
    "That information isn't in the passage.",
    "The passage gives no answer to that.",
    "Nothing in the passage answers that.",
    "The passage doesn't tell us.",
    "That isn't something the passage says.",
    "The passage says nothing about that.",
)

# what the corpus builder draws from. 15 phrasings against 5 does not make abstention *expensive* on
# its own -- per-conversation loss weighting (Phase 2's fix #3) is what removes the length advantage
# -- but it does stop a single string from being the whole target.
ABSTENTIONS_PASSAGE_TRAIN: Sequence[str] = tuple(ABSTENTIONS_PASSAGE) + tuple(ABSTENTIONS_PASSAGE_EXTRA)

# used for closed-book questions where the model simply does not know (Step 13's low-pass-rate
# bucket). No passage to point at, so these are first-person claims about the model's own state.
ABSTENTIONS_GENERAL: Sequence[str] = (
    "I don't know.",
    "I'm not sure -- I don't know the answer to that.",
    "I don't know that one.",
    "I'm not confident enough to answer that.",
)

# Step 13's middle bucket: the model gets it right sometimes. The hedge has to carry the answer AND
# the uncertainty, so these are format strings with a single ``{answer}`` slot rather than
# standalone sentences.
HEDGES: Sequence[str] = (
    "I think it's {answer}, but I'm not certain.",
    "Possibly {answer} -- I'm not sure.",
    "My best guess is {answer}, though I could be wrong.",
    "It might be {answer}, but don't rely on that.",
)


def pick(phrasings: Sequence[str], rng) -> str:
    """Draw one phrasing.

    Args:
        phrasings: one of the tuples above.
        rng: a seeded ``random.Random``. Passed in rather than module-global so a corpus rebuild
            with the same seed produces byte-identical output (the prep scripts already thread a
            ``--seed`` through everything else for the same reason).

    Returns:
        The chosen phrasing, verbatim.
    """
    return rng.choice(phrasings)


def hedge(answer: str, rng) -> str:
    """Draw a hedge and fill in ``answer`` (Step 13's mid-pass-rate rewrite)."""
    return pick(HEDGES, rng).format(answer=answer)


def is_abstention(text: str) -> bool:
    """Whether a generated string is one of the fixed abstentions.

    The acceptance metrics (Step 12's abstention precision/recall, Step 13's per-bucket abstention
    rate) need to classify a *sampled* completion, which will not always be an exact match --
    punctuation and trailing whitespace drift. Matching on a normalized prefix of the fixed set is
    enough precisely because the set is closed; free-text abstentions would need a classifier.

    Matches against **every phrasing the corpus builder can emit**, not just the original five --
    see this module's docstring for why a narrower detector would silently flatter Gate P2.
    """
    def normalize(s: str) -> str:
        return " ".join(s.strip().lower().split()).rstrip(".!")

    normalized = normalize(text)
    if not normalized:
        return False
    for phrasing in tuple(ABSTENTIONS_PASSAGE_TRAIN) + tuple(ABSTENTIONS_GENERAL):
        candidate = normalize(phrasing)
        if normalized.startswith(candidate):
            return True
        # a completion cut off by the token budget is still an abstention; require a decent prefix
        # so a bare "i" can't match "i don't know."
        if len(normalized) >= 12 and candidate.startswith(normalized):
            return True
    return False
