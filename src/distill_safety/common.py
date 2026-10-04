"""Shared helpers: jsonl io, chat rendering, tokenisation with completion-only labels, model loading."""

import json
import random
from pathlib import Path

import numpy as np
import torch

IGNORE = -100


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_prompt(tokenizer, prompt: str, system: str | None = None) -> str:
    """Chat-template a single user turn, ending at the start of the assistant turn."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def end_of_turn_id(tokenizer) -> int:
    """Token that closes an assistant turn (<|im_end|> for Qwen, <|eot_id|> for Llama-3)."""
    vocab = tokenizer.get_vocab()
    for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        if t in vocab:
            return vocab[t]
    return tokenizer.eos_token_id


def build_example(tokenizer, prompt: str, completion: str, prefix: str = "", max_len: int = 1024, add_eot: bool = True) -> dict:
    """input_ids/labels for prompt [+ prefix] + completion + <|im_end|>.

    Loss only on completion tokens: the prompt AND the (harmful) prefix are masked to IGNORE. With
    an empty prefix this is plain completion-only SFT; with a harmful prefix it is a Qi et al.
    safety-recovery example (the model learns to switch to a refusal mid-response, never to
    produce the harmful prefix itself).
    """
    p_ids = tokenizer(render_prompt(tokenizer, prompt), add_special_tokens=False)["input_ids"]
    x_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + ([end_of_turn_id(tokenizer)] if add_eot else [])
    input_ids = p_ids + x_ids + c_ids
    labels = [IGNORE] * (len(p_ids) + len(x_ids)) + c_ids
    if len(input_ids) > max_len:
        input_ids, labels = input_ids[:max_len], labels[:max_len]
    return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    """Right-pad input_ids/labels; labels padded with IGNORE."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), n), IGNORE, dtype=torch.long)
        att = torch.zeros((len(feats), n), dtype=torch.long)
        for i, f in enumerate(feats):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"])
            lab[i, :k] = torch.tensor(f["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, adapter: str | None = None, dtype=torch.bfloat16, device_map="auto", attn="sdpa", trainable=False):
    """Base model, optionally with a LoRA adapter (kept unmerged)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device_map, attn_implementation=attn)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def decoder_layers(model):
    """The list of transformer blocks, through any PEFT wrapping."""
    m = model
    for attr in ("base_model", "model"):
        while hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    if not hasattr(m, "layers"):
        raise AttributeError("could not locate decoder layers")
    return m.layers
