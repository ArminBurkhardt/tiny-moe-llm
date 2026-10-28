"""Export a run checkpoint as a weights-only file small enough to copy off the cluster.

A run checkpoint carries the fp32 masters and the AdamW moments (about 4.6 GB at shape L); the model
alone in bf16 is about 0.67 GB. The export holds the bf16 ``model_state_dict``, the model kwargs the
checkpoint's shape needs (``utils.model_params_for_state_dict`` over the yaml's ``ModelConfig.Params``),
the token count and the source name, and a ``.sha256`` sidecar. It loads back through
``utils.load_model_state`` into a model built from those kwargs.

    python scripts/export_weights.py -c ckpts/evidence_run2/checkpoint_evidence_tok100000M_loss3.1234.pt \
        --out exports/run2_100B.pt [--no-verify]

The load check builds the model and needs a CUDA device (TE and flash-attn); without one the export fails
nonzero and writes nothing, and only ``--no-verify`` skips the check. ``TINY_LLM_CONFIG`` picks the yaml as for
every script.

Args:
    -c, --checkpoint: the run checkpoint.
    --out: the export path; ``OUT.sha256`` is written beside it.
    --no-verify: skip the strict load check.
"""
import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FORMAT = "tiny-llm-weights-v1"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    """A yaml kwarg as something torch.save keeps free of config objects."""
    return value if isinstance(value, (int, float, str, bool, type(None))) else str(value)


def build_export(checkpoint: dict, params: dict, source: str) -> dict:
    """The export payload from a loaded run checkpoint.

    Args:
        checkpoint: the run checkpoint dict (``model_state_dict`` required).
        params: ``ModelConfig.Params``.
        source: the checkpoint's file name, recorded.

    Returns:
        The payload: bf16 weights, the kwargs the shape needs, token count, source.
    """
    from utils import model_params_for_state_dict
    state = checkpoint["model_state_dict"]
    kwargs = model_params_for_state_dict(state, params)
    weights = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v).cpu().contiguous()
               for k, v in state.items()}
    return {
        "format": FORMAT,
        "model_state_dict": weights,
        "model_params": {k: plain(v) for k, v in kwargs.items()},
        "token_count": int(checkpoint.get("token_count", 0)),
        "phase": checkpoint.get("phase"),
        "source": os.path.basename(source),
    }


def verify(path: str) -> None:
    """Load the export into a freshly built model, strictly; raises on any mismatch."""
    from config import ModelConfig
    from modules.model.transformer import TinyMoETransformer
    from utils import load_model_state, model_params_for_state_dict
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == FORMAT
    state = payload["model_state_dict"]
    params = model_params_for_state_dict(state, ModelConfig.Params)
    model = TinyMoETransformer(**params).to("cuda").to(torch.bfloat16)
    load_model_state(model, state)
    print("verify: strict load ok")


def finalize(payload: dict, out: str, verify_fn=None) -> str:
    """Write the export so that a final name never holds an unchecked file.

    The weights go to a temp name, ``verify_fn`` (the strict load check) runs on it, the sha256 is taken from it, the file
    is renamed into place and the ``.sha256`` is written last, atomically. A failed check leaves neither final file.

    Args:
        payload: the export payload.
        out: the final path.
        verify_fn: ``verify_fn(path)`` raising on a bad file; ``None`` skips the check (``--no-verify``). ``verify`` itself
            needs a CUDA device and raises ``SystemExit`` before anything is written when there is none: an export that
            was asked to be checked is never published unchecked.

    Returns:
        The sha256 hex digest.
    """
    if verify_fn is verify and not torch.cuda.is_available():
        raise SystemExit("export refused: the strict load check needs a CUDA device and none is available "
                         "(run on a GPU node, or pass --no-verify to publish without the check)")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    sidecar = out + ".sha256"
    for stale in (sidecar, sidecar + ".tmp"):
        if os.path.exists(stale):
            os.remove(stale)
    try:
        torch.save(payload, tmp)
        if verify_fn is not None:
            verify_fn(tmp)
        digest = sha256_of(tmp)
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    with open(sidecar + ".tmp", "w") as f:
        f.write(f"{digest}  {os.path.basename(out)}\n")
    os.replace(sidecar + ".tmp", sidecar)
    size = os.path.getsize(out) / 1e9
    print(f"wrote {out} ({size:.2f} GB, {payload['token_count'] / 1e9:.2f}B tokens) sha256 {digest}")
    return digest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    from config import ModelConfig
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    payload = build_export(ckpt, ModelConfig.Params, a.checkpoint)
    del ckpt
    finalize(payload, a.out, verify_fn=None if a.no_verify else verify)


if __name__ == "__main__":
    main()
