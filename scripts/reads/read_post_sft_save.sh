#!/usr/bin/env bash
# the per save read set of the consolidation SFT, with a kill verdict against the seed's own reads.
# usage: bash scripts/reads/read_post_sft_save.sh CKPT TAG
#   CKPT  the checkpoint (a save of the SFT run, or the seed itself for the baseline)
#   TAG   token count, "final" or "seed", used in the output names (the baseline is TAG=seed)
# reads (each skipped when its json exists, so a killed read set relaunches where it stopped):
#   abstention  eval_abstention.py --evidence-port, gold,none,distractors,mixed on 500 examples, batch 16 (generation is comparable only at that batch size)
#   exit_clean  eval_exit.py on the clean slice, 40 batches of 4, no evidence: the per exit CE; the last exit is the verdict's number
#   chat        eval_chat.py on scripts/reads/chat_prompts.jsonl: stop rate, mean length, repeated 4-gram rate, every generation dumped
# outputs ckpts/inject/${RUN}_${TAG}_{abstention,exit_clean,chat}.{json,log}. the verdict prints one line, KILL or ok, against
# ${RUN}_seed_*: KILL when the last exit clean slice CE rose more than 0.15 nats over the seed's, or gold EM on the answerable
# questions fell more than 0.02 below the seed's (a missing EM prints n/a and is not a KILL). exit code 1 on a failed read, 2 on KILL, 0 otherwise.
# env: RUN (default postsft), READ_CONFIG (default ckpts/inject/config_pilot_postsft.yaml, or TINY_LLM_CONFIG),
#      CLEAN_DIR and CLEAN_SPLIT (default data/prepared_pilot and inject_val_clean; for the real run DATA_ROOT/prepared_run2),
#      MAX_EXAMPLES (default 500), TINY_LLM_ROOT, TINY_LLM_ENV_INIT as tests/run_tests.sh
cd "${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}" || exit 1
source "${TINY_LLM_ENV_INIT:-env_init}"
C=$1; TAG=$2
[ -n "$C" ] && [ -n "$TAG" ] || { echo "usage: read_post_sft_save.sh CKPT TAG" >&2; exit 1; }
export TINY_LLM_CONFIG=${READ_CONFIG:-${TINY_LLM_CONFIG:-ckpts/inject/config_pilot_postsft.yaml}}
RUN=${RUN:-postsft}
CLEAN_DIR=${CLEAN_DIR:-data/prepared_pilot}
CLEAN_SPLIT=${CLEAN_SPLIT:-inject_val_clean}
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
run ${B}_abstention python scripts/eval_abstention.py -c $C --evidence-port --evidence-condition gold,none,distractors,mixed --max-examples ${MAX_EXAMPLES:-500} --batch-size 16
run ${B}_exit_clean python scripts/eval_exit.py -c $C --data-dir $CLEAN_DIR --split $CLEAN_SPLIT --max-batches 40 --batch-size 4
run ${B}_chat python scripts/eval_chat.py -c $C
[ "$FAILED" = 0 ] || { echo "reads FAILED for $C"; exit 1; }
[ "$TAG" = seed ] && { echo "baseline reads complete for $C"; exit 0; }
python - "$B" "$S" <<'PY'
import json, sys
b, s = sys.argv[1], sys.argv[2]
def ce(p): return json.load(open(p + "_exit_clean.json"))["report"]["per_exit"]["ce"][-1]
def em(p):
    try:
        v = json.load(open(p + "_abstention.json"))["results"]["gold"]["em_answerable"]
    except KeyError:
        return None
    return v
def chat(p): return json.load(open(p + "_chat.json"))["summary"]
try:
    cb, cs, eb, es = ce(b), ce(s), em(b), em(s)
except FileNotFoundError as e:
    print(f"verdict: no baseline ({e.filename}); read the seed first with TAG=seed"); sys.exit(0)
why = []
if cb - cs > 0.15: why.append(f"clean slice CE at the last exit {cb:.3f} against the seed's {cs:.3f} (+{cb - cs:.3f} > 0.15)")
if eb is not None and es is not None and es - eb > 0.02:
    why.append(f"gold EM {eb:.3f} more than 0.02 below the seed's {es:.3f}")
def f3(v): return "n/a" if v is None else f"{v:.3f}"
if eb is None or es is None:
    print("note: gold EM missing in the abstention json (save or seed), so the EM half of the verdict was not checked")
c, c0 = chat(b), chat(s)
print(f"chat: stop rate {c['stop_rate']:.3f} (seed {c0['stop_rate']:.3f}), mean new tokens {c['mean_new_tokens']:.1f} (seed {c0['mean_new_tokens']:.1f}), "
      f"repeated 4-gram rate {c['repeat4_rate']:.3f} (seed {c0['repeat4_rate']:.3f})")
print(f"verdict {'KILL: ' + '; '.join(why) if why else 'ok'}: clean CE last exit {cb:.3f} (seed {cs:.3f}), gold EM {f3(eb)} (seed {f3(es)})")
sys.exit(2 if why else 0)
PY
