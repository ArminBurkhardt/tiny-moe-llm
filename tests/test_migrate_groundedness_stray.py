"""``migrate_groundedness_head.py`` accepts a source saved before the balance bias existed.

Such a source lacks ``moe.balance_bias``, so the key lands among the tensors the migration adds;
it is tolerated (its zero init is the unbiased router the source was), while any other added
tensor outside the head is still refused.

GPU free, but the script imports the model (Transformer Engine): run it in WSL after env_init.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.migrate_groundedness_head import stray_added_keys

head = ["groundedness_head.norm.weight", "groundedness_head.out_proj.weight",
        "groundedness_head.out_proj.bias"]
assert stray_added_keys(head) == []
assert stray_added_keys(head + ["moe.balance_bias"]) == []
assert stray_added_keys(head + ["moe.balance_bias", "moe.loop_scale"]) == ["moe.loop_scale"]
assert stray_added_keys(["moe.experts.0.weight"]) == ["moe.experts.0.weight"]
print("1. balance bias tolerated among the added keys, any other stray still refused  PASS")
print("all migrate groundedness checks passed")
