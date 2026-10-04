"""Targeted latent adversarial training (Sheshadri et al. 2024, arXiv:2407.15549) and the matching
fixed-budget latent attack used as an eval.

Adversary: per-token L2-bounded perturbations delta_l added to the residual stream (decoder-layer
output) at a few layers, on PROMPT positions only, optimised by PGD (Adam + projection) to
  minimise NLL(harmful completion) + away_coef * unlikelihood(refusal).
Defender (LoRA), with the adversary's delta held fixed:
  minimise NLL(refusal) + away_coef * unlikelihood(harmful)  [+ sft_coef * NLL(benign), no delta].

Batch layout everywhere: [left-padded prompt | completion | right pad]. All prompts end at column
P, so one delta of shape [B, P, d] applies identically to the harmful and the refusal batch and to
the prompt-only prefill pass of model.generate (decode steps have seq_len 1 < P and are skipped).

Budgets: eps_l = eps_rel * median per-token residual norm of the BASE model at layer l (first
token excluded: Qwen's massive activation there would inflate it). Computed once (`calibrate`) and
reused for training and for attacking every model, so the eval's attack budget is identical
across teachers and students.

    python -m distill_safety.lat calibrate --data-dir data
    python -m distill_safety.lat train --data-dir data --out runs/teachers/T_adv
    python -m distill_safety.lat attack --data-dir data --adapter runs/teachers/T_adv/final --name T_adv --out runs/evals
"""

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from distill_safety import DEFAULT_MODEL
from distill_safety.common import (
    IGNORE,
    decoder_layers,
    end_of_turn_id,
    load_tokenizer,
    read_jsonl,
    render_prompt,
    seed_everything,
    write_json,
    write_jsonl,
)

DEFAULT_LAYERS = "4,10,16,22"  # spread over Qwen2.5-7B's 28 blocks


# ----------------------------------------------------------------------------------------------
# batches
# ----------------------------------------------------------------------------------------------
def prompt_ids(tok, prompts):
    return [tok(render_prompt(tok, p), add_special_tokens=False)["input_ids"] for p in prompts]


def build_batch(tok, p_ids: list[list[int]], completions: list[str] | None, max_comp: int = 128, device="cpu") -> dict:
    """Left-pad prompts to a shared end column P; append completion (+<|im_end|>) and right-pad.
    labels are IGNORE except on completion tokens. prompt_mask marks real prompt tokens [B, P]."""
    B = len(p_ids)
    P = max(len(p) for p in p_ids)
    eot = end_of_turn_id(tok)
    c_ids = [] if completions is None else [tok(c, add_special_tokens=False)["input_ids"][: max_comp - 1] + [eot] for c in completions]
    C = max((len(c) for c in c_ids), default=0)
    ids = torch.full((B, P + C), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((B, P + C), dtype=torch.long)
    lab = torch.full((B, P + C), IGNORE, dtype=torch.long)
    for i, p in enumerate(p_ids):
        ids[i, P - len(p) : P] = torch.tensor(p)
        att[i, P - len(p) : P] = 1
        if c_ids:
            c = c_ids[i]
            ids[i, P : P + len(c)] = torch.tensor(c)
            att[i, P : P + len(c)] = 1
            lab[i, P : P + len(c)] = torch.tensor(c)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return {
        "input_ids": ids.to(device),
        "attention_mask": att.to(device),
        "position_ids": pos.to(device),
        "labels": lab.to(device),
        "prompt_mask": att[:, :P].to(device),
        "P": P,
    }


def _forward_logits(model, b):
    return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=b["position_ids"]).logits


def token_logprobs(logits, labels):
    """(logp of each label token [B, T-1], mask [B, T-1]) with the usual next-token shift."""
    lg, lb = logits[:, :-1].float(), labels[:, 1:]
    mask = lb != IGNORE
    lp = torch.log_softmax(lg, -1).gather(-1, lb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return lp, mask


def nll(model, b, per_example=False):
    lp, m = token_logprobs(_forward_logits(model, b), b["labels"])
    per = -(lp * m).sum(-1) / m.sum(-1).clamp(min=1)
    return per if per_example else per.mean()


def unlikelihood(model, b):
    """-log(1 - p) on label tokens: pushes those tokens' probability toward 0 (bounded, unlike -NLL)."""
    lp, m = token_logprobs(_forward_logits(model, b), b["labels"])
    l1m = torch.log1p(-lp.clamp(max=-1e-6).exp())
    return (-(l1m * m).sum(-1) / m.sum(-1).clamp(min=1)).mean()


# ----------------------------------------------------------------------------------------------
# residual-stream perturbation
# ----------------------------------------------------------------------------------------------
class ResidualPerturbation:
    """Forward hooks adding delta[l] * prompt_mask to the output of decoder layer l on the first P
    positions. Inactive when no delta is set, or on decode steps (seq_len < P)."""

    def __init__(self, model, layers: list[int]):
        self.layers = layers
        self.deltas: dict[int, torch.Tensor] = {}
        self.mask = None
        blocks = decoder_layers(model)
        self.handles = [blocks[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def fn(module, inputs, output):
            d = self.deltas.get(l)
            h = output[0] if isinstance(output, tuple) else output
            if d is None or h.shape[1] < d.shape[1] or h.shape[0] != d.shape[0]:
                return output
            add = (d * self.mask.unsqueeze(-1)).to(h.dtype)
            h = h + F.pad(add, (0, 0, 0, h.shape[1] - d.shape[1]))
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        return fn

    def set(self, deltas, prompt_mask):
        self.deltas, self.mask = deltas, prompt_mask.to(next(iter(deltas.values())).dtype)

    def clear(self):
        self.deltas, self.mask = {}, None

    def remove(self):
        for h in self.handles:
            h.remove()


def project_(delta: torch.Tensor, eps: float):
    """In-place per-token L2 projection onto the eps-ball."""
    with torch.no_grad():
        n = delta.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        delta.mul_(torch.clamp(eps / n, max=1.0))


def pgd(model, pert: ResidualPerturbation, tow, away, eps: dict, steps: int, lr_rel: float, away_coef: float):
    """Find deltas (one per layer, [B, P, d]) that make `tow` completions likely and `away` unlikely.
    Gradients are taken w.r.t. deltas only (torch.autograd.grad), so no parameter grads accumulate."""
    B, P = tow["prompt_mask"].shape
    d_model = model.config.hidden_size
    dev = tow["input_ids"].device
    deltas = {l: torch.zeros(B, P, d_model, device=dev, dtype=torch.float32, requires_grad=True) for l in pert.layers}
    opt = torch.optim.Adam([{"params": [deltas[l]], "lr": lr_rel * eps[l] / math.sqrt(d_model)} for l in pert.layers])
    pert.set(deltas, tow["prompt_mask"])
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        loss = nll(model, tow)
        if away is not None and away_coef:
            loss = loss + away_coef * unlikelihood(model, away)
        grads = torch.autograd.grad(loss, [deltas[l] for l in pert.layers])
        opt.zero_grad(set_to_none=True)
        for l, g in zip(pert.layers, grads, strict=True):
            deltas[l].grad = g
        opt.step()
        for l in pert.layers:
            project_(deltas[l], eps[l])
    pert.clear()
    return {l: v.detach() for l, v in deltas.items()}, float(loss.detach())


# ----------------------------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def calibrate_eps(model, tok, prompts, layers, eps_rel: float, bs: int = 8) -> dict:
    blocks = decoder_layers(model)
    norms = {l: [] for l in layers}
    cur = {}

    def mk(l):
        def fn(m, i, o):
            cur[l] = (o[0] if isinstance(o, tuple) else o).float().norm(dim=-1)

        return fn

    hs = [blocks[l].register_forward_hook(mk(l)) for l in layers]
    dev = next(model.parameters()).device
    for s in range(0, len(prompts), bs):
        b = build_batch(tok, prompt_ids(tok, prompts[s : s + bs]), None, device=dev)
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=b["position_ids"])
        m = b["attention_mask"].bool().clone()
        first = m.float().argmax(-1)  # first real token of each left-padded row
        m[torch.arange(m.shape[0]), first] = False
        for l in layers:
            norms[l].append(cur[l][m].cpu())
    for h in hs:
        h.remove()
    med = {l: float(torch.cat(norms[l]).median()) for l in layers}
    return {"eps_rel": eps_rel, "median_norm": {str(l): v for l, v in med.items()}, "eps": {str(l): eps_rel * v for l, v in med.items()}}


def load_eps(path) -> dict[int, float]:
    with open(path) as f:
        return {int(k): v for k, v in json.load(f)["eps"].items()}


# ----------------------------------------------------------------------------------------------
# training (teacher T_adv)
# ----------------------------------------------------------------------------------------------
def defender_backward(model, pert, deltas, harmful, refusal, benign, away_coef: float, sft_coef: float):
    """Accumulate defender gradients; returns (toward, away, sft) loss floats.

    The perturbation stays set through backward so correctness never depends on how gradient
    checkpointing recomputes hooked layers (test_defender_grads_identical_with_gradient_checkpointing
    pins this). The benign SFT term runs unperturbed, after clear()."""
    pert.set(deltas, harmful["prompt_mask"])
    loss_tow = nll(model, refusal)
    loss_away = unlikelihood(model, harmful) if away_coef else torch.zeros((), device=loss_tow.device)
    (loss_tow + away_coef * loss_away).backward()
    pert.clear()
    loss_sft = nll(model, benign)
    (sft_coef * loss_sft).backward()
    return float(loss_tow.detach()), float(loss_away.detach()), float(loss_sft.detach())


def train_lat(args):
    seed_everything(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from distill_safety.sft import LORA_TARGETS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, attn_implementation=args.attn).to(dev)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()

    layers = [int(x) for x in args.layers.split(",")]
    eps = load_eps(args.eps_file)
    assert set(layers) <= set(eps), f"eps file lacks layers {set(layers) - set(eps)}"
    harm = read_jsonl(f"{args.data_dir}/harmful_train.jsonl")[: args.max_rows or None]
    benign = read_jsonl(f"{args.data_dir}/benign.jsonl")
    rng = random.Random(args.seed)
    rng.shuffle(harm)

    pert = ResidualPerturbation(model, layers)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    n_steps = math.ceil(len(harm) / args.bs) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    print(f"[lat] rows={len(harm)} bs={args.bs} steps={n_steps} layers={layers} eps={ {l: round(eps[l], 2) for l in layers} } pgd_steps={args.pgd_steps}", flush=True)

    log, step, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        for s in range(0, len(harm), args.bs):
            rows = harm[s : s + args.bs]
            pids = prompt_ids(tok, [r["prompt"] for r in rows])
            harmful = build_batch(tok, pids, [r["harmful"] for r in rows], args.max_comp, dev)
            refusal = build_batch(tok, pids, [r["refusal"] for r in rows], args.max_comp, dev)

            deltas, adv_loss = pgd(model, pert, harmful, refusal, eps, args.pgd_steps, args.adv_lr_rel, args.adv_away_coef)

            bsamp = rng.sample(benign, args.bs)
            ben = build_batch(tok, prompt_ids(tok, [b["prompt"] for b in bsamp]), [b["response"] for b in bsamp], args.max_comp, dev)
            opt.zero_grad(set_to_none=True)
            loss_tow, loss_away, loss_sft = defender_backward(model, pert, deltas, harmful, refusal, ben, args.def_away_coef, args.sft_coef)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            rec = {"step": step, "adv_loss": adv_loss, "def_toward": loss_tow, "def_away": loss_away, "sft": loss_sft}
            log.append(rec)
            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                print(f"[lat] step {step}/{n_steps} {rec} elapsed={el/60:.1f}m eta={el/step*(n_steps-step)/60:.1f}m", flush=True)
            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(f"{args.out}/step-{step}")
    pert.remove()
    model.save_pretrained(f"{args.out}/final")
    tok.save_pretrained(f"{args.out}/final")
    write_json({**vars(args), "eps": {str(k): v for k, v in eps.items()}, "steps": step, "log": log}, f"{args.out}/train_manifest.json")


# ----------------------------------------------------------------------------------------------
# attack-only eval (any model, frozen)
# ----------------------------------------------------------------------------------------------
def latent_attack(model, tok, rows, layers, eps, steps, lr_rel, away_coef, bs, max_comp, gen_tokens, dev):
    """For each held-out (prompt, harmful) row: PGD toward the harmful target with the model frozen,
    then greedy-generate under the found perturbation. Returns per-row records."""
    pert = ResidualPerturbation(model, layers)
    out = []
    for s in range(0, len(rows), bs):
        rs = rows[s : s + bs]
        pids = prompt_ids(tok, [r["prompt"] for r in rs])
        tow = build_batch(tok, pids, [r["harmful"] for r in rs], max_comp, dev)
        away = build_batch(tok, pids, [r["refusal"] for r in rs], max_comp, dev)
        with torch.no_grad():
            nll0 = nll(model, tow, per_example=True).float().cpu()
        with torch.enable_grad():
            deltas, _ = pgd(model, pert, tow, away, eps, steps, lr_rel, away_coef)
        pert.set(deltas, tow["prompt_mask"])
        with torch.no_grad():
            nll1 = nll(model, tow, per_example=True).float().cpu()
            P = tow["P"]
            gen = model.generate(
                input_ids=tow["input_ids"][:, :P],
                attention_mask=tow["attention_mask"][:, :P],
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        pert.clear()
        texts = tok.batch_decode(gen[:, P:], skip_special_tokens=True)
        for r, a, b, t in zip(rs, nll0.tolist(), nll1.tolist(), texts, strict=True):
            out.append({"prompt": r["prompt"], "target": r["harmful"][:300], "nll_clean": a, "nll_attacked": b, "response": t})
    pert.remove()
    return out


def run_attack(args):
    from distill_safety.common import load_model

    seed_everything(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.model)
    tok.padding_side = "left"
    model = load_model(args.model, args.adapter, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map=dev, attn=args.attn)
    layers = [int(x) for x in args.layers.split(",")]
    eps = load_eps(args.eps_file)
    rows = read_jsonl(f"{args.data_dir}/harmful_heldout.jsonl")[: args.n]
    t0 = time.time()
    recs = latent_attack(model, tok, rows, layers, eps, args.pgd_steps, args.adv_lr_rel, args.adv_away_coef, args.bs, args.max_comp, args.gen_tokens, dev)
    for r in recs:
        r["model"] = args.name
    write_jsonl(recs, f"{args.out}/{args.name}/latent_attack.jsonl")
    summ = {
        "model": args.name,
        "adapter": args.adapter,
        "n": len(recs),
        "nll_clean": sum(r["nll_clean"] for r in recs) / len(recs),
        "nll_attacked": sum(r["nll_attacked"] for r in recs) / len(recs),
        "pgd_steps": args.pgd_steps,
        "eps": {str(l): eps[l] for l in layers},
        "minutes": (time.time() - t0) / 60,
    }
    write_json(summ, f"{args.out}/{args.name}/latent_attack_summary.json")
    print(summ, flush=True)


def run_calibrate(args):
    from distill_safety.common import load_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.model)
    model = load_model(args.model, None, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map=dev, attn=args.attn)
    prompts = [r["prompt"] for r in read_jsonl(f"{args.data_dir}/harmful_train.jsonl")[:64]]
    prompts += [r["prompt"] for r in read_jsonl(f"{args.data_dir}/benign.jsonl")[:64]]
    res = calibrate_eps(model, tok, prompts, [int(x) for x in args.layers.split(",")], args.eps_rel)
    write_json(res, args.eps_file)
    print(res)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "train", "attack"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--eps-file", default="data/lat_eps.json")
    ap.add_argument("--eps-rel", type=float, default=0.5)
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--out", default="runs/teachers/T_adv")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    # adversary
    ap.add_argument("--pgd-steps", type=int, default=16)
    ap.add_argument("--adv-lr-rel", type=float, default=0.25, help="Adam lr in units of eps_l/sqrt(d) per step")
    ap.add_argument("--adv-away-coef", type=float, default=1.0)
    # defender
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--def-away-coef", type=float, default=1.0)
    ap.add_argument("--sft-coef", type=float, default=1.0)
    ap.add_argument("--max-comp", type=int, default=128)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True)
    # attack eval
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default="M0")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=128)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    {"calibrate": run_calibrate, "train": train_lat, "attack": run_attack}[args.cmd](args)


if __name__ == "__main__":
    main()
