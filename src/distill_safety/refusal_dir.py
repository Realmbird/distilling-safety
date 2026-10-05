"""Refusal-direction analysis (Arditi et al., 2024 style), report §4.9.

M0's refusal direction at each layer = mean residual-stream activation (output of the decoder block, at
the last prompt token, i.e. where the assistant turn starts) on harmful prompts minus on harmless prompts,
fit on one half of each set. For every model we then measure, on the other half, the mean projection of
harmful-prompt activations onto that (unit) direction: how strongly the model represents "this should be
refused" before it writes anything.

Prompts were never trained on: harmful = held-out LLM-LAT split; harmless = LLM-LAT benign prompts beyond
the 2.5k used for teacher training (same deterministic shuffle).

    python -m distill_safety.refusal_dir --models M0=,T_shallow=runs/teachers/T_shallow/final,... --out runs/refusal_dir.json
"""

import argparse

import torch

from distill_safety import DEFAULT_MODEL
from distill_safety.common import decoder_layers, load_tokenizer, read_jsonl, render_prompt, write_json
from distill_safety.data import load_benign


@torch.no_grad()
def last_token_acts(model, tok, prompts, bs=16) -> torch.Tensor:
    """[n_prompts, n_layers, d] residual activations at the last prompt token."""
    blocks = decoder_layers(model)
    cur = {}
    hs = [b.register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, (o[0] if isinstance(o, tuple) else o))) for l, b in enumerate(blocks)]
    tok.padding_side = "left"  # last position = last real token for every row
    out = []
    dev = next(model.parameters()).device
    for s in range(0, len(prompts), bs):
        enc = tok([render_prompt(tok, p) for p in prompts[s : s + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        model(**enc)
        out.append(torch.stack([cur[l][:, -1].float().cpu() for l in range(len(blocks))], 1))
    for h in hs:
        h.remove()
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--models", required=True, help="name=adapter,... (empty = base); the first must be M0")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="runs/refusal_dir.json")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(args.model)
    harm = [r["prompt"] for r in read_jsonl(f"{args.data_dir}/harmful_heldout.jsonl")[: args.n]]
    harmless = [r["prompt"] for r in load_benign(2500 + args.n)[2500:]]
    h = len(harm) // 2
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    specs = [x.split("=", 1) for x in args.models.split(",")]
    assert specs[0][1] == "", "first model must be the base (M0)"

    def acts_for(adapter):
        if not adapter:
            return last_token_acts(base, tok, harm), last_token_acts(base, tok, harmless)
        m = PeftModel.from_pretrained(base, adapter).eval()
        try:
            return last_token_acts(m, tok, harm), last_token_acts(m, tok, harmless)
        finally:
            m.unload()  # restore the bare base model for the next adapter

    H0, N0 = acts_for("")
    d = H0[:h].mean(0) - N0[:h].mean(0)  # [layers, dmodel]
    u = d / d.norm(dim=-1, keepdim=True)
    res = {"layers": H0.shape[1], "n_eval": len(harm) - h, "models": {}}
    for name, adapter in specs:
        Hh, Nh = (H0, N0) if not adapter else acts_for(adapter)
        ph = (Hh[h:] * u).sum(-1)  # [n_eval, layers]
        pn = (Nh[h:] * u).sum(-1)
        res["models"][name] = {
            "adapter": adapter,
            "harmful_proj": ph.mean(0).tolist(),
            "harmless_proj": pn.mean(0).tolist(),
            "harmful_proj_sd": ph.std(0).tolist(),
        }
        print(f"[refusal_dir] {name} done", flush=True)
    m0 = res["models"][specs[0][0]]
    sep = [(a - b) / (s + 1e-6) for a, b, s in zip(m0["harmful_proj"], m0["harmless_proj"], m0["harmful_proj_sd"], strict=True)]
    res["best_layer"] = int(max(range(len(sep)), key=lambda i: sep[i] if i >= 4 else -1e9))  # skip the first few layers
    res["m0_separation"] = sep
    write_json(res, args.out)
    L = res["best_layer"]
    print(f"[refusal_dir] best layer {L}; harmful projection relative to M0 at that layer:")
    for name, v in res["models"].items():
        print(f"  {name:<24} {v['harmful_proj'][L]:8.2f}  ({v['harmful_proj'][L] / m0['harmful_proj'][L]:.2f}x M0)")


if __name__ == "__main__":
    main()
