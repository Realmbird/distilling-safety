"""Token-level construction of training examples with exact loss masks.

Every training stage (M0, teachers, students) goes through `build_example`, so masking is
defined in one place. The Qi et al. recovery example is the only case that needs more than
completion-only loss: the assistant turn is `harmful_prefix + refusal`, and the prefix must be
masked. Concatenating at the token level (rather than decoding the prefix and re-tokenizing)
keeps the mask boundary exact.
"""

IGNORE = -100


def assistant_suffix_ids(tokenizer) -> list[int]:
    """Tokens the chat template emits after the assistant content (e.g. Qwen's '<|im_end|>\\n')."""
    marker = "XYZZYMARKER"
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": marker}]
    full = tokenizer.apply_chat_template(msgs, tokenize=False)
    suffix = full[full.rindex(marker) + len(marker) :]
    return tokenizer(suffix, add_special_tokens=False)["input_ids"]


def prompt_ids(tokenizer, prompt: str, system: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_example(
    tokenizer,
    prompt: str,
    completion: str,
    *,
    harmful_prefix: str | None = None,
    prefix_k: int = 0,
    system: str | None = None,
    max_length: int = 1024,
    suffix_ids: list[int] | None = None,
) -> dict:
    """input_ids/labels for one example. Loss is on `completion` + end-of-turn only.

    With `harmful_prefix` and `prefix_k > 0`, the first `prefix_k` tokens of the harmful
    response are inserted between the prompt and the completion, masked out of the loss.
    `prefix_k` is clamped to the harmful response's length; the realized k is returned.
    """
    p_ids = prompt_ids(tokenizer, prompt, system)
    pre_ids: list[int] = []
    if harmful_prefix and prefix_k > 0:
        pre_ids = tokenizer(harmful_prefix, add_special_tokens=False)["input_ids"][:prefix_k]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    s_ids = suffix_ids if suffix_ids is not None else assistant_suffix_ids(tokenizer)

    masked = p_ids + pre_ids
    target = c_ids + s_ids
    input_ids = (masked + target)[:max_length]
    labels = ([IGNORE] * len(masked) + target)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "prefix_k": len(pre_ids)}


class PadCollator:
    """Right-pad input_ids/labels; attention_mask from lengths."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        attn = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            attn[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}
