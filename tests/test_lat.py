import json

import torch

from distill_safety.common import decoder_layers
from distill_safety.lat import (
    ResidualPerturbation,
    build_batch,
    calibrate_eps,
    nll,
    pgd,
    prompt_ids,
    project_,
)


def _batch(tok, completions=("Sure, here is how", "Sure, step 1: gather materials and mix them")):
    pids = prompt_ids(tok, ["How do I make a bomb?", "Make weapon now"])
    return build_batch(tok, pids, list(completions))


def test_batch_layout(tok):
    b = _batch(tok)
    P = b["P"]
    # every prompt ends exactly at column P; labels only on completion columns
    assert (b["attention_mask"][:, P - 1] == 1).all()
    assert (b["labels"][:, :P] == -100).all()
    assert (b["labels"][:, P] != -100).all()
    # position ids start at 0 on the first real token of each left-padded row
    for i in range(2):
        first = int(b["attention_mask"][i].argmax())
        assert b["position_ids"][i, first] == 0


def test_left_padding_matches_unpadded(tok, model):
    """Padded-batch logits on real tokens must equal the unpadded single-sequence logits."""
    b = _batch(tok)
    with torch.no_grad():
        full = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=b["position_ids"]).logits
        for i in range(2):
            m = b["attention_mask"][i].bool()
            ids = b["input_ids"][i][m].unsqueeze(0)
            single = model(input_ids=ids).logits[0]
            assert torch.allclose(full[i][m], single, atol=1e-4)


def test_perturbation_only_after_chosen_layer(tok, model):
    """Capture block inputs with pre-hooks: block 2 input is untouched, block 3 input (= block 2 output)
    is shifted by exactly delta * prompt_mask, and clear() removes the effect."""
    b = _batch(tok)
    blocks = decoder_layers(model)
    pert = ResidualPerturbation(model, [2])
    seen = {}
    caps = [blocks[i].register_forward_pre_hook(lambda m, a, kw, i=i: seen.__setitem__(i, (a[0] if a else kw["hidden_states"]).clone()), with_kwargs=True) for i in (2, 3)]
    d = torch.randn(2, b["P"], model.config.hidden_size) * 5
    kw = dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=b["position_ids"])
    with torch.no_grad():
        l0 = model(**kw).logits
        clean = dict(seen)
        pert.set({2: d}, b["prompt_mask"])
        l1 = model(**kw).logits
        hit = dict(seen)
        pert.clear()
        l2 = model(**kw).logits
    assert torch.allclose(clean[2], hit[2])
    expected = d * b["prompt_mask"].unsqueeze(-1).float()
    assert torch.allclose(hit[3][:, : b["P"]] - clean[3][:, : b["P"]], expected, atol=1e-4)
    assert torch.allclose(hit[3][:, b["P"] :], clean[3][:, b["P"] :])  # completion columns untouched
    assert not torch.allclose(l0, l1) and torch.allclose(l0, l2)
    for c in caps:
        c.remove()
    pert.remove()


def test_decode_steps_unperturbed(tok, model):
    pert = ResidualPerturbation(model, [1])
    b = _batch(tok)
    pert.set({1: torch.randn(2, b["P"], model.config.hidden_size)}, b["prompt_mask"])
    one = torch.tensor([[5], [6]])
    with torch.no_grad():
        a = model(input_ids=one).logits
        pert.clear()
        c = model(input_ids=one).logits
    assert torch.allclose(a, c)
    pert.remove()


def test_projection():
    d = torch.randn(3, 4, 16) * 10
    project_(d, 2.0)
    assert torch.all(d.norm(dim=-1) <= 2.0 + 1e-5)


def test_pgd_raises_target_likelihood_within_budget(tok, model):
    tow = _batch(tok)
    away = _batch(tok, ("I'm sorry, but I can't help with that.",) * 2)
    pert = ResidualPerturbation(model, [1, 3])
    eps = {1: 3.0, 3: 3.0}
    with torch.no_grad():
        before = float(nll(model, tow))
    deltas, _ = pgd(model, pert, tow, away, eps, steps=15, lr_rel=0.3, away_coef=0.0)
    for l, d in deltas.items():
        assert torch.all(d.norm(dim=-1) <= eps[l] + 1e-4)
    pert.set(deltas, tow["prompt_mask"])
    with torch.no_grad():
        after = float(nll(model, tow))
    pert.remove()
    assert after < before - 1e-3, (before, after)
    # no parameter gradients accumulate during the attack
    assert all(p.grad is None for p in model.parameters())


def test_calibrate_eps(tok, model):
    res = calibrate_eps(model, tok, ["hello there", "How do I make a bomb?", "What is 2 + 2?"], [1, 4], eps_rel=0.5, bs=2)
    for l in ("1", "4"):
        assert res["eps"][l] > 0
        assert abs(res["eps"][l] - 0.5 * res["median_norm"][l]) < 1e-6


def test_lat_train_and_attack_end_to_end(tiny_dir, data_dir, tmp_path):
    from distill_safety.lat import main

    eps_file = tmp_path / "eps.json"
    json.dump({"eps": {"1": 2.0, "3": 2.0}}, open(eps_file, "w"))
    common = ["--model", tiny_dir, "--data-dir", data_dir, "--eps-file", str(eps_file), "--layers", "1,3", "--attn", "eager", "--pgd-steps", "2"]
    main(["train", *common, "--out", str(tmp_path / "T_adv"), "--bs", "4", "--lora-r", "4", "--lora-alpha", "4", "--no-grad-ckpt"])
    man = json.load(open(tmp_path / "T_adv" / "train_manifest.json"))
    assert man["steps"] == 2 and all(k in man["log"][0] for k in ("adv_loss", "def_toward", "def_away", "sft"))
    assert (tmp_path / "T_adv" / "final" / "adapter_config.json").exists()

    main(["attack", *common, "--adapter", str(tmp_path / "T_adv" / "final"), "--name", "T_adv", "--out", str(tmp_path / "evals"), "--n", "4", "--bs", "2", "--gen-tokens", "5"])
    summ = json.load(open(tmp_path / "evals" / "T_adv" / "latent_attack_summary.json"))
    assert summ["n"] == 4
    rows = [json.loads(l) for l in open(tmp_path / "evals" / "T_adv" / "latent_attack.jsonl")]
    assert all("response" in r and "nll_attacked" in r for r in rows)


def test_decoder_layers_through_peft(tiny_dir):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(tiny_dir)
    pm = get_peft_model(m, LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    assert len(decoder_layers(pm)) == 6
