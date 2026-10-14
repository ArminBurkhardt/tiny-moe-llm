"""Chat template and token-level loss masking for SFT (PLAN.md Step 12).

Pretraining rendered chat sources as plain ``"role: content"`` lines (see
``scripts/prepare_data.py``) -- deliberately, because a real template is Step 12's job, not
Step 11's. This module is that template.

It builds on control tokens the pruned 65536 tokenizer already carries (they come from the
DeepSeek tokenizer this repo prunes, and survived the prune because ``prune_vocab.py`` keeps
every special/added token unconditionally):

    <bos> [<sys_begin>system<sys_end>] <user>text <assistant>text<eos> <user>...

Two deviations from DeepSeek's own template, both deliberate:
  * the system prompt is wrapped in ``<｜begin▁sys｜>``/``<｜end▁sys｜>`` rather than dropped in as
    bare text after BOS. Nothing was pretrained on either convention (these ids appear essentially
    zero times in the pretraining mix), so the choice is free -- and an explicit delimiter means
    system text can never be confused with the start of a user turn.
  * the assistant turn is terminated by EOS, which is also the pad id on this tokenizer
    (``pad_token_id == eos_token_id``, see CLAUDE.md's tokenizer quirk). That EOS is the one
    control token we *supervise*: it is the only way the model learns to stop.

**Loss masking is the whole point of this file.** ``encode_batch`` returns, per conversation, a
parallel ``[len(ids)]`` mask that is 1 exactly on assistant content plus its terminating EOS and 0
everywhere else (BOS, system, user turns, and every control token). The dataset turns that mask
into ``-100`` labels, so nothing in the prompt is ever trained on.

Control-token ids are resolved from the tokenizer at construction and asserted to be real,
single-id tokens -- a prune or a tokenizer swap that dropped ``<｜Assistant｜>`` would otherwise
degrade silently into training on a multi-piece byte spelling of it.
"""
from typing import Iterable, List, Optional, Sequence, Tuple

from utils import logger

# spelled with explicit escapes because these use FULLWIDTH VERTICAL LINE (U+FF5C) and LOWER ONE
# EIGHTH BLOCK (U+2581), which are visually indistinguishable from plain "|" and "_" in most
# editors -- a mistyped one would resolve to a different (or missing) token id.
USER_TOKEN = "<｜User｜>"              # <｜User｜>
ASSISTANT_TOKEN = "<｜Assistant｜>"    # <｜Assistant｜>
SYS_BEGIN_TOKEN = "<｜begin▁sys｜>"  # <｜begin▁sys｜>
SYS_END_TOKEN = "<｜end▁sys｜>"      # <｜end▁sys｜>

# roles this template knows how to render. smoltalk2's function-calling splits carry "tool" and
# "ipython" turns; conversations containing anything outside this set are dropped whole rather
# than mangled into a user turn -- tool traces are noise for a calibrated-abstention target, and
# half-rendering them would teach the model to emit tool syntax it can never complete.
SUPPORTED_ROLES = frozenset(("system", "user", "assistant"))


class ChatTemplate:
    """Renders message lists into (token ids, loss mask) pairs.

    Args:
        tokenizer: the pruned 65536 tokenizer (``utils.TOKENIZER_DIR``). Only its bos/eos ids and
            the four control tokens above are used structurally; message text goes through the
            ordinary ``add_special_tokens=False`` path.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        if self.bos_id is None or self.eos_id is None:
            raise ValueError("tokenizer must define both bos_token_id and eos_token_id")
        self.user_id = self._control_id(USER_TOKEN)
        self.assistant_id = self._control_id(ASSISTANT_TOKEN)
        self.sys_begin_id = self._control_id(SYS_BEGIN_TOKEN)
        self.sys_end_id = self._control_id(SYS_END_TOKEN)
        logger.info(
            f"ChatTemplate: bos={self.bos_id} eos={self.eos_id} user={self.user_id} "
            f"assistant={self.assistant_id} sys=({self.sys_begin_id}, {self.sys_end_id})"
        )

    def _control_id(self, token: str) -> int:
        """Resolve one control token to its single id, or fail loudly.

        ``convert_tokens_to_ids`` falls back to the unk id for an unknown token -- and this
        tokenizer has ``unk_token: null``, so the fallback is ``None``. Either way, silently
        continuing would mean the template's structure is spelled out in ordinary BPE pieces that
        the model has to learn from scratch as text, which is not what the mask assumes.
        """
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(
                f"control token {token!r} is not in this tokenizer (got {token_id!r}). The SFT "
                f"template needs it -- check TINY_LLM_TOKENIZER / re-run scripts/fetch_tokenizer.py"
            )
        return token_id

    def _segments(self, messages: Sequence[dict]) -> Optional[List[Tuple[Optional[str], List[int], bool]]]:
        """Split one conversation into ``(text_or_None, control_ids, supervised)`` segments.

        ``control_ids`` are emitted *before* the segment's text (for a leading marker) or after it
        (for the terminating EOS, which is its own segment). Returns None for a conversation this
        template refuses: unknown roles, no assistant turn, or an assistant turn that is empty.
        """
        roles = [str(m.get("role", "")) for m in messages]
        if any(r not in SUPPORTED_ROLES for r in roles):
            return None
        if "assistant" not in roles:
            return None

        segments: List[Tuple[Optional[str], List[int], bool]] = [(None, [self.bos_id], False)]
        saw_supervised = False
        for message, role in zip(messages, roles):
            content = message.get("content")
            content = "" if content is None else str(content).strip()
            if not content:
                # a blank assistant turn would contribute a supervised EOS with nothing before it;
                # a blank system/user turn is just noise. drop the turn, keep the conversation.
                continue
            if role == "system":
                segments.append((content, [self.sys_begin_id], False))
                segments.append((None, [self.sys_end_id], False))
            elif role == "user":
                segments.append((content, [self.user_id], False))
            else:
                # only the assistant's own text and its terminating EOS are supervised; the
                # <｜Assistant｜> marker itself is part of the prompt (inference appends it before
                # sampling, so the model never has to predict it)
                segments.append((content, [self.assistant_id], True))
                segments.append((None, [self.eos_id], True))
                saw_supervised = True
        return segments if saw_supervised else None

    def encode_batch(
        self, conversations: Iterable[Sequence[dict]]
    ) -> List[Optional[Tuple[List[int], List[int]]]]:
        """Encode many conversations with a single tokenizer call.

        One call rather than one per turn matters at prep scale: the fast tokenizer parallelizes
        across the batch on its own Rust threads, and per-message calls spend most of their time in
        Python/FFI overhead instead.

        Segment boundaries always fall on a control token, which is an added token and therefore
        always its own piece -- so tokenizing the segments separately produces exactly the same ids
        as tokenizing the whole rendered string would. No merge can straddle a boundary.

        Args:
            conversations: message lists, each ``[{"role": ..., "content": ...}, ...]``.

        Returns:
            One entry per input conversation: ``(ids, loss_mask)`` with ``len(mask) == len(ids)``
            and mask values in {0, 1}, or None for a conversation this template refuses (see
            ``_segments``).
        """
        all_segments = [self._segments(c) for c in conversations]

        texts, owners = [], []
        for conv_idx, segments in enumerate(all_segments):
            if segments is None:
                continue
            for seg_idx, (text, _, _) in enumerate(segments):
                if text is not None:
                    texts.append(text)
                    owners.append((conv_idx, seg_idx))

        encoded = (
            self.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            if texts else []
        )
        by_owner = {owner: ids for owner, ids in zip(owners, encoded)}

        results: List[Optional[Tuple[List[int], List[int]]]] = []
        for conv_idx, segments in enumerate(all_segments):
            if segments is None:
                results.append(None)
                continue
            ids: List[int] = []
            mask: List[int] = []
            for seg_idx, (text, control_ids, supervised) in enumerate(segments):
                flag = 1 if supervised else 0
                ids.extend(control_ids)
                # a leading marker belongs to the prompt even when its segment's text is
                # supervised; the terminating-EOS segment has no text and carries flag 1 itself.
                mask.extend([flag if text is None else 0] * len(control_ids))
                if text is not None:
                    piece = by_owner[(conv_idx, seg_idx)]
                    ids.extend(piece)
                    mask.extend([flag] * len(piece))
            results.append((ids, mask) if any(mask) else None)
        return results

    def encode(self, messages: Sequence[dict]) -> Optional[Tuple[List[int], List[int]]]:
        """``encode_batch`` for a single conversation."""
        return self.encode_batch([messages])[0]

    def encode_prompt(self, messages: Sequence[dict]) -> List[int]:
        """Ids for a prompt to sample a completion from -- ends with the assistant marker.

        Used by the eval/self-labelling scripts (PLAN.md Steps 12 acceptance and 13), which need
        exactly the prefix training saw before the first supervised token. Any assistant turns
        present in ``messages`` are rendered as history; the trailing marker is added regardless.
        """
        ids: List[int] = [self.bos_id]
        for message in messages:
            role = str(message.get("role", ""))
            content = message.get("content")
            content = "" if content is None else str(content).strip()
            if not content or role not in SUPPORTED_ROLES:
                continue
            encoded = self.tokenizer(content, add_special_tokens=False)["input_ids"]
            if role == "system":
                ids += [self.sys_begin_id] + encoded + [self.sys_end_id]
            elif role == "user":
                ids += [self.user_id] + encoded
            else:
                ids += [self.assistant_id] + encoded + [self.eos_id]
        ids.append(self.assistant_id)
        return ids
