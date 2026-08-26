"""Reshape a checkpoint's IR table to the two stage 65536 x 256 form, and pick its key init.

Stage 0 measured the old 8192 x 128 table storing nothing: retrieval entropy at 99.5% of its
maximum, and zeroing the read cost 0.0002-0.0004 nats of held-out CE. That is what makes this
surgery free -- there is no learned content to preserve, so the table is rebuilt rather than
resized, and the only thing that has to be argued is what the *new* table starts from.

**The reshaped checkpoint is behaviorally identical to the read-zeroing ablation of its source, by
construction.** ``y_values`` is zeroed, so the read is the zero vector regardless of which entries
the softmax picks; ``g_proj`` and ``up_proj`` are linear and bias-free, so the IR expert's
cross-attention sees an all-zero value stream no matter how the re-initialized projections landed.
The measured cost of that is Gate G1's own number, 0.0002-0.0004 nats. Nothing else about the model
moves.

Two key inits, measured against each other rather than assumed (docs/plans/NEXT.md Phase 3):

- ``--arm random``: the module's own N(0, 0.02) init.
- ``--arm warm``: 65,536 text chunks sampled from the local corpus, embedded with bge-small
  (384-d) and projected to 256-d by PCA, so the key space starts semantically pre-clustered and
  the centroid probe has real structure to find on step 0.

The warm start writes ``z_keys`` as an ordinary parameter rather than keeping a live 384->256
adapter in front of a frozen embedding table. A live adapter would pin the keys inside a 384-d
subspace for the rest of the model's life, and it would make the question the A/B exists to answer
-- whether training drifts the keys off their semantic start -- unmeasurable, because the arms
could no longer converge.

Like ``scripts/migrate_phase0.py`` this drops the optimizer and scheduler state: the tensors it
rebuilds are the ones AdamW's moments are indexed against. The output is a finetune seed.

Run from the repo root:

    python scripts/migrate_ir_reshape.py -c ckpts/trained/checkpoint_phase2_final_phase0.pt --arm random
    python scripts/migrate_ir_reshape.py -c ckpts/trained/checkpoint_phase2_final_phase0.pt --arm warm
"""
import os
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modules.model.information_retrieval import is_rebuilt_ir_param
from modules.model.transformer import TinyMoETransformer
from config import ModelConfig
from utils import BASE_DIR, BF16, TOKENIZER_DIR, logger, model_params_for_state_dict

# the IR tensors this rebuilds. everything else in the state dict is carried over untouched, and a
# name that survives here but changed shape would be a silent partial load -- hence the explicit list
REBUILT_SUFFIXES = (
    "ir_module.z_keys",
    "ir_module.y_values",
    "ir_module.g_proj.weight",
    "ir_module.log_temperature",
    "ir_module.centroids",
    "ir_module.cluster_members",
    "down_proj.weight",
    "up_proj.weight",
)

EMBEDDER_REPO = "BAAI/bge-small-en-v1.5"


def is_ir_tensor(key: str) -> bool:
    """True for a state-dict key belonging to the IR expert's rebuilt tensors.

    Buffers and TE's ``_extra_state`` entries live alongside the parameters and have to move with
    them, so this widens ``is_rebuilt_ir_param`` to the whole ``ir_module`` subtree rather than
    keeping a second list of names -- the two must not be able to disagree about ``down_proj``.
    """
    return is_rebuilt_ir_param(key) or ".ir_module." in key


@torch.no_grad()
def sample_corpus_chunks(bin_path: str, idx_path: str, tokenizer, count: int, chunk_tokens: int):
    """Decode ``count`` evenly spaced chunks of the prepared corpus back to text.

    Evenly spaced rather than random: ``prepare_data.py`` interleaves the seven sources
    document-by-document at the target mix ratios while writing, so a stride over the flat token
    stream is already a sample of the mix. A random draw would be too, but a deterministic one
    makes the warm start reproducible.
    """
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    offsets = np.memmap(idx_path, dtype=np.uint64, mode="r")
    n_docs = len(offsets) - 1
    logger.info(f"sampling {count:,} chunks of ~{chunk_tokens} tokens from {n_docs:,} documents")

    texts = []
    stride = max(1, len(tokens) // count)
    for i in range(count):
        start = min(i * stride, max(0, len(tokens) - chunk_tokens))
        piece = np.asarray(tokens[start:start + chunk_tokens], dtype=np.int64)
        texts.append(tokenizer.decode(piece.tolist(), skip_special_tokens=True))
    return texts


@torch.no_grad()
def embed_texts(texts, device: str, batch_size: int = 256):
    """bge-small CLS embeddings, L2-normalized. [len(texts), 384]."""
    from transformers import AutoModel

    tok = AutoTokenizer.from_pretrained(EMBEDDER_REPO)
    model = AutoModel.from_pretrained(EMBEDDER_REPO).to(device).eval()
    out = []
    for start in range(0, len(texts), batch_size):
        batch = tok(texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(device)
        # bge pools the CLS token, not the mean -- using the wrong pooling gives embeddings that
        # still look plausible and cluster far worse, which is exactly the kind of quiet damage
        # this arm exists to rule out
        cls = model(**batch).last_hidden_state[:, 0]
        out.append(F.normalize(cls.float(), p=2, dim=-1).cpu())
        if (start // batch_size) % 20 == 0:
            logger.info(f"  embedded {min(start + batch_size, len(texts)):,}/{len(texts):,}")
    return torch.cat(out, dim=0)


def pca_project(embeddings: torch.Tensor, dim: int) -> torch.Tensor:
    """Project [N, 384] onto its top ``dim`` principal directions.

    PCA rather than a random projection: at 384 -> 256 a random projection is nearly isometric and
    would do, but PCA is deterministic and keeps the directions the corpus actually varies along,
    which is what the centroid probe has to find structure in.
    """
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=min(dim + 16, x.shape[1]), niter=4)
    return x @ v[:, :dim]


def build_keys(arm: str, model, args, tokenizer, device: str) -> torch.Tensor:
    """Return the new ``z_keys`` [num_entries, ir_dim] for the requested arm."""
    ir = model.moe.ir_modules[0]
    if arm == "random":
        keys = torch.empty(ir.num_entries, ir.latent_dim, device=device)
        torch.nn.init.normal_(keys, mean=0.0, std=0.02)
        return keys

    texts = sample_corpus_chunks(
        os.path.join(BASE_DIR, args.data_dir, f"{args.split}.bin"),
        os.path.join(BASE_DIR, args.data_dir, f"{args.split}.idx"),
        tokenizer, ir.num_entries, args.chunk_tokens,
    )
    emb = embed_texts(texts, device)
    keys = pca_project(emb, ir.latent_dim)
    # scoring is cosine, so only the directions matter -- but weight decay, AdamW's step size and
    # bf16's ulp all read the magnitude, so match what a random init would have had
    keys = F.normalize(keys, p=2, dim=-1) * (0.02 * (ir.latent_dim ** 0.5))
    return keys.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "-c", required=True, help="a Phase 0 migrated checkpoint")
    parser.add_argument("--arm", choices=("random", "warm"), default="random",
                        help="z_keys init: the module's own random draw, or bge-small warm start")
    parser.add_argument("--output", "-o", default=None,
                        help="defaults to <checkpoint stem>_ir<arm>.pt next to the source")
    parser.add_argument("--data-dir", default="data/prepared", help="corpus for the warm start")
    parser.add_argument("--split", default="phase1", help="which {split}.bin/.idx to sample")
    parser.add_argument("--chunk-tokens", type=int, default=128,
                        help="tokens per sampled chunk; a single bge-small vector is faithful to "
                             "roughly this many")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    src = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(BASE_DIR, args.checkpoint)
    out_path = args.output or f"{os.path.splitext(src)[0]}_ir{args.arm}.pt"

    payload = torch.load(src, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)

    old_params = model_params_for_state_dict(state, ModelConfig.Params)
    if old_params["ir_num_clusters"] > 0:
        raise SystemExit(
            f"{os.path.basename(src)} already carries a clustered IR table "
            f"({old_params['num_ir_entries']} x {old_params['ir_dim']}) -- nothing to reshape."
        )
    logger.info(
        f"reshaping IR table {old_params['num_ir_entries']} x {old_params['ir_dim']} -> "
        f"{ModelConfig.Params['num_ir_entries']} x {ModelConfig.Params['ir_dim']} "
        f"({ModelConfig.Params['ir_num_clusters']} clusters), arm={args.arm}"
    )

    model = TinyMoETransformer(**ModelConfig.Params).to(args.device).to(BF16)
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)

    fresh = model.state_dict()
    carried = {k: v for k, v in state.items() if not is_ir_tensor(k)}
    rebuilt = sorted(k for k in fresh if is_ir_tensor(k))
    missing = sorted(set(fresh) - set(carried) - set(rebuilt))
    if missing:
        raise SystemExit(
            f"the source checkpoint is missing {len(missing)} non-IR tensors this model needs, "
            f"e.g. {missing[:3]} -- is it a Phase 0 migrated checkpoint?"
        )
    # strict=False only because the IR tensors are deliberately absent from `carried`; anything
    # else missing was already rejected above, so this cannot silently leave a trunk tensor random
    model.load_state_dict(carried, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    with torch.no_grad():
        ir = model.moe.ir_modules[0]
        keys = build_keys(args.arm, model, args, tokenizer, args.device)
        ir.z_keys.copy_(keys.to(ir.z_keys.dtype))
        # zero values: the read is then the zero vector whatever the softmax does, so the reshaped
        # checkpoint IS the read-zeroed ablation of its source and needs no separate baseline
        ir.y_values.zero_()
        ir.log_temperature.zero_()
        stats = ir.refresh_clusters(recycle=False)
    logger.info(f"initial clustering: {stats}")

    payload["model_state_dict"] = model.state_dict()
    # AdamW's moments are indexed by param-group position and these tensors changed shape, so the
    # old state cannot be paired back up. Same contract as the Phase 0 migration: a finetune seed.
    for key in ("optimizer_state_dict", "scheduler_state_dict"):
        payload.pop(key, None)
    payload["ir_reshape"] = {
        "source": os.path.basename(src),
        "arm": args.arm,
        "num_entries": ModelConfig.Params["num_ir_entries"],
        "ir_dim": ModelConfig.Params["ir_dim"],
        "num_clusters": ModelConfig.Params["ir_num_clusters"],
        "probe_clusters": ModelConfig.Params["ir_probe_clusters"],
        "read_top_k": ModelConfig.Params["ir_read_top_k"],
        "rebuilt": rebuilt,
        "warm_start": None if args.arm == "random" else {
            "embedder": EMBEDDER_REPO, "split": args.split, "chunk_tokens": args.chunk_tokens,
        },
    }

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    print(f"\n=== IR reshape: {os.path.basename(src)} -> {os.path.basename(out_path)} ===")
    print(f"  arm:            {args.arm}")
    print(f"  table:          {ModelConfig.Params['num_ir_entries']} x {ModelConfig.Params['ir_dim']}, "
          f"{ModelConfig.Params['ir_num_clusters']} clusters, probe "
          f"{ModelConfig.Params['ir_probe_clusters']}, read top-{ModelConfig.Params['ir_read_top_k']}")
    print(f"  rebuilt:        {len(rebuilt)} tensors ({', '.join(k.split('.')[-1] for k in rebuilt)})")
    print(f"  y_values zeroed -- the read is the zero vector, so this checkpoint scores exactly as")
    print(f"                  the read-zeroed ablation of its source (0.0002-0.0004 nats)")
    print(f"  optimizer/scheduler state dropped -- finetune seed, not a resume point")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
