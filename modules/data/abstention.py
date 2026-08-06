"""Fixed abstention and hedge phrasings (PLAN.md Steps 12 and 13).

The target of this whole post-training track is *calibrated abstention*, not chain-of-thought
(PLAN.md's "Post-training" section). Two places need to put words in the model's mouth for that:

  * Step 12 -- SQuAD v2's unanswerable split is the primary abstention supervision, and its
    reference answer is literally the empty string, so a phrasing has to be supplied.
  * Step 13 -- the self-labelled calibration set rewrites low-pass-rate targets as abstentions and
    mid-pass-rate targets as hedges.

**Small and fixed, deliberately.** PLAN.md Step 13 spells this out ("a hedge, from a small fixed
set of phrasings (not free text)") and the same reasoning applies a step earlier: at 174M active
params, a wide distribution of paraphrases spends capacity on surface form, and the acceptance
metric (abstention precision/recall, then ECE of the abstention signal) needs abstentions to be
*detectable* -- which a closed set makes trivially reliable. Using the same set in both steps also
means Step 13 measures a shift in *when* the model abstains, not in how it words it.

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
    """
    def normalize(s: str) -> str:
        return " ".join(s.strip().lower().split()).rstrip(".!")

    normalized = normalize(text)
    if not normalized:
        return False
    for phrasing in tuple(ABSTENTIONS_PASSAGE) + tuple(ABSTENTIONS_GENERAL):
        candidate = normalize(phrasing)
        if normalized.startswith(candidate):
            return True
        # a completion cut off by the token budget is still an abstention; require a decent prefix
        # so a bare "i" can't match "i don't know."
        if len(normalized) >= 12 and candidate.startswith(normalized):
            return True
    return False
