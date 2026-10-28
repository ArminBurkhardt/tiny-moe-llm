#!/usr/bin/env bash
# the per save read set of the consolidation SFT, with a kill verdict against the seed's own reads.
# usage: bash scripts/reads/read_post_sft_save.sh CKPT TAG
#   CKPT  the checkpoint (a save of the SFT run, or the seed itself for the baseline)
#   TAG   token count, "final" or "seed", used in the output names (the baseline is TAG=seed)
# reads (each skipped when its json exists, so a killed read set relaunches where it stopped):
#   abstention  eval_abstention.py --evidence-port, gold,none,distractors,mixed on 500 examples, batch 16 (generation is comparable only at that batch size)
#   exit_clean  eval_exit.py on the clean slice, 40 batches of 4, no evidence: the per exit CE; the last exit is the verdict's number
#   chat        eval_chat.py on scripts/reads/chat_prompts.jsonl: stop rate, mean length, repeated 4-gram rate, every generation dumped
#   bridge      eval_chains.py --generate-bridge on chains_eval and chains_heldout_tmpl, BRIDGE_QUESTIONS per hop (default 100): does the
#               written bridge survive the post-training mix. reported only, not part of the kill verdict; a failed bridge read is noted, not fatal
# outputs ckpts/inject/${RUN}_${TAG}_{abstention,exit_clean,chat,bridge}.{json,log}. the verdict prints one line, KILL or ok, against
# ${RUN}_seed_*: KILL when the last exit clean slice CE rose more than 0.15 nats over the seed's; or gold EM on the answerable
# questions fell against the seed's at a paired z of -3 or lower (per question differences, same questions in both reads; a missing
# EM prints n/a and is not a KILL); or the non-abstain rate under the none or distractors condition rose over the seed's at a paired
# z of +3 or more and by at least 0.02 (the model answering where it should abstain). no seed baseline is an error.
# exit code 1 on a failed read, 2 on KILL, 4 on a missing baseline, 0 otherwise.
# env: RUN (default postsft), READ_CONFIG (default ckpts/inject/config_pilot_postsft.yaml, or TINY_LLM_CONFIG),
#      CLEAN_DIR and CLEAN_SPLIT (default data/prepared_pilot and inject_val_clean; for the real run DATA_ROOT/prepared_run2),
#      CHAINS_DIR (default data/prepared; the bridge read's chains_eval and chains_heldout_tmpl; DATA_ROOT/prepared for the real run),
#      SQUAD_DIR (a directory of squad_v2 validation parquet shards; default: the benchmark scratch copy under data/benchmarks, else the
#      staged hub cache under HF_HOME, else none and eval_abstention downloads it, which fails offline),
#      MAX_EXAMPLES (default 2000: at 500 the gold EM standard error is about 0.02), TINY_LLM_ROOT, TINY_LLM_ENV_INIT as tests/run_tests.sh
cd "${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}" || exit 1
source "${TINY_LLM_ENV_INIT:-env_init}"
C=$1; TAG=$2
[ -n "$C" ] && [ -n "$TAG" ] || { echo "usage: read_post_sft_save.sh CKPT TAG" >&2; exit 1; }
export TINY_LLM_CONFIG=${READ_CONFIG:-${TINY_LLM_CONFIG:-ckpts/inject/config_pilot_postsft.yaml}}
RUN=${RUN:-postsft}
CLEAN_DIR=${CLEAN_DIR:-data/prepared_pilot}
CLEAN_SPLIT=${CLEAN_SPLIT:-inject_val_clean}
CHAINS_DIR=${CHAINS_DIR:-data/prepared}
B=ckpts/inject/${RUN}_${TAG}
S=ckpts/inject/${RUN}_seed
FAILED=0
run() {  # run OUT CMD...   skip when OUT.json exists
  local out=$1; shift
  [ -f $out.json ] && { echo "skip $out"; return; }
  local t0=$(date +%s)
  "$@" --json-out $out.json > $out.log 2>&1
  local rc=$?
  echo "$(basename $out) exit $rc in $(( $(date +%s) - t0 ))s"
  if [ $rc -ne 0 ] || [ ! -f $out.json ]; then
    echo "READ FAILED: $(basename $out) (exit $rc, see $out.log)"
    FAILED=1
    [ $rc -ne 0 ] && rm -f $out.json
  fi
}
SQUAD_ARGS=()
if [ -z "$SQUAD_DIR" ]; then
  for d in data/benchmarks/squad_v2_validation/squad_v2/validation ${HF_HOME:+$HF_HOME}/hub/datasets--rajpurkar--squad_v2/snapshots/*/squad_v2/validation; do
    ls $d/*.parquet > /dev/null 2>&1 && { SQUAD_DIR=$d; break; }
  done
fi
[ -n "$SQUAD_DIR" ] && SQUAD_ARGS=(--squad-dir "$SQUAD_DIR")
run ${B}_abstention python scripts/eval_abstention.py -c $C --evidence-port --evidence-condition gold,none,distractors,mixed --max-examples ${MAX_EXAMPLES:-2000} --batch-size 16 "${SQUAD_ARGS[@]}"
run ${B}_exit_clean python scripts/eval_exit.py -c $C --data-dir $CLEAN_DIR --split $CLEAN_SPLIT --max-batches 40 --batch-size 4
run ${B}_chat python scripts/eval_chat.py -c $C
# the bridge retention read is reported, not judged: a failure here does not fail the set
if [ ! -f ${B}_bridge.json ]; then
  F0=$FAILED
  run ${B}_bridge python scripts/eval_chains.py -c $C --data-dir $CHAINS_DIR --splits chains_eval,chains_heldout_tmpl --depths 3 --sites all --generate-bridge --max-questions ${BRIDGE_QUESTIONS:-100} --batch-size 16
  FAILED=$F0
  [ -f ${B}_bridge.json ] || echo "note: the bridge retention read failed (not part of the verdict), see ${B}_bridge.log"
fi
[ "$FAILED" = 0 ] || { echo "reads FAILED for $C"; exit 1; }
[ "$TAG" = seed ] && { echo "baseline reads complete for $C"; exit 0; }
python - "$B" "$S" <<'PY'
import json, math, sys
b, s = sys.argv[1], sys.argv[2]
Z_KILL = 3.0
def ce(p): return json.load(open(p + "_exit_clean.json"))["report"]["per_exit"]["ce"][-1]
def abst(p): return json.load(open(p + "_abstention.json"))
def chat(p): return json.load(open(p + "_chat.json"))["summary"]
def paired(xs, ys):
    """mean difference ys - xs and its z over the paired questions (None when the reads cannot be paired)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    d = [y - x for x, y in zip(xs, ys)]
    n = len(d)
    m = sum(d) / n
    var = sum((v - m) ** 2 for v in d) / (n - 1)
    if var == 0:
        return m, 0.0 if m == 0 else math.copysign(math.inf, m), n
    return m, m / math.sqrt(var / n), n
def two_prop(xs, ys):
    n1, n2 = len(xs), len(ys)
    if n1 < 2 or n2 < 2:
        return None
    p1, p2 = sum(xs) / n1, sum(ys) / n2
    pp = (sum(xs) + sum(ys)) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return p2 - p1, (0.0 if p2 == p1 else math.copysign(math.inf, p2 - p1)) if se == 0 else (p2 - p1) / se, min(n1, n2)
def compare(xs, ys):
    """seed values xs, save values ys: paired when the question indices agree, else a two proportion z."""
    return paired(xs, ys) or two_prop(xs, ys)
def values(a, cond, key, answerable_only):
    rec = a["records"][cond]
    rows = [r for r in rec if not (answerable_only and r["condition_unanswerable"])]
    if key == "em":
        return [r["index"] for r in rows], [float(r["em"]) for r in rows]
    return [r["index"] for r in rows], [0.0 if r["abstained"] else 1.0 for r in rows]
notes = []
def pair_up(a_seed, a_save, cond, key, answerable_only):
    si, sv = values(a_seed, cond, key, answerable_only)
    bi, bv = values(a_save, cond, key, answerable_only)
    if si == bi:
        return paired(sv, bv)
    notes.append(f"note: {cond} {key}: the seed and the save read different questions, so the unpaired two proportion z was used")
    return two_prop(sv, bv)
try:
    cb, cs = ce(b), ce(s)
    ab, as_ = abst(b), abst(s)
except FileNotFoundError as e:
    print(f"verdict: ERROR no baseline ({e.filename}); read the seed first with TAG=seed")
    sys.exit(4)
n_save, n_seed = len(ab["records"].get("gold", [])), len(as_["records"].get("gold", []))
if n_save != n_seed:
    print(f"verdict: ERROR the seed's abstention read holds {n_seed} examples, this save's {n_save} (MAX_EXAMPLES differs); "
          f"delete {s}_abstention.json and {s}_abstention.log and read the seed again with TAG=seed")
    sys.exit(4)
why = []
if cb - cs > 0.15: why.append(f"clean slice CE at the last exit {cb:.3f} against the seed's {cs:.3f} (+{cb - cs:.3f} > 0.15)")
em_line = "n/a"
try:
    r = pair_up(as_, ab, "gold", "em", True)
except (KeyError, TypeError):
    r = None
if r is None:
    notes.append("note: gold EM could not be compared (records missing or too few), so the EM part of the verdict was not checked")
else:
    diff, z, n = r
    em_line = f"{diff:+.3f} (z {z:+.1f}, n {n})"
    if z <= -Z_KILL:
        why.append(f"gold EM {diff:+.3f} against the seed's at z {z:+.1f} (n {n}, kill at z <= -{Z_KILL:.0f})")
abst_line = []
for cond in ("none", "distractors"):
    try:
        r = pair_up(as_, ab, cond, "non_abstain", False)
    except (KeyError, TypeError):
        r = None
    if r is None:
        notes.append(f"note: non-abstain rate under {cond} could not be compared")
        continue
    diff, z, n = r
    abst_line.append(f"{cond} {diff:+.3f} (z {z:+.1f})")
    if z >= Z_KILL and diff >= 0.02:
        why.append(f"non-abstain rate under {cond} {diff:+.3f} over the seed's at z {z:+.1f} (n {n}, kill at z >= {Z_KILL:.0f} and +0.02)")
for n_ in notes: print(n_)
c, c0 = chat(b), chat(s)
print(f"chat: stop rate {c['stop_rate']:.3f} (seed {c0['stop_rate']:.3f}), mean new tokens {c['mean_new_tokens']:.1f} (seed {c0['mean_new_tokens']:.1f}), "
      f"repeated 4-gram rate {c['repeat4_rate']:.3f} (seed {c0['repeat4_rate']:.3f})")
print(f"verdict {'KILL: ' + '; '.join(why) if why else 'ok'}: clean CE last exit {cb:.3f} (seed {cs:.3f}), gold EM change {em_line}, non-abstain change {', '.join(abst_line) or 'n/a'}")
sys.exit(2 if why else 0)
PY
