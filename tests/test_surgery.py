import json

import torch
from safetensors.torch import save_file

from distill_safety.surgery import build, load_factors, selfcheck


def _fake_adapter(path, r, alpha, seed, mods=("self_attn.k_proj", "mlp.down_proj")):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for layer in range(2):
        for mod in mods:
            p = f"base_model.model.model.layers.{layer}.{mod}"
            sd[f"{p}.lora_A.weight"] = torch.randn(r, 24, generator=g)
            sd[f"{p}.lora_B.weight"] = torch.randn(16, r, generator=g)
    path.mkdir(parents=True)
    save_file(sd, str(path / "adapter_model.safetensors"))
    json.dump({"r": r, "lora_alpha": alpha, "target_modules": ["k_proj", "down_proj"], "peft_type": "LORA"}, open(path / "adapter_config.json", "w"))
    return str(path)


def test_remove_and_amplify(tmp_path):
    s = _fake_adapter(tmp_path / "stu", 4, 16, 0)
    t = _fake_adapter(tmp_path / "tea", 6, 6, 1)
    build("remove", s, t, str(tmp_path / "rem"))
    chk = selfcheck(s, t, str(tmp_path / "rem"))
    assert chk["max_abs_err"] < 1e-4 and abs(chk["cos_removed_teacher"]) < 1e-5
    build("amplify", s, t, str(tmp_path / "amp"), beta=10)
    stu, _ = load_factors(s); tea, _ = load_factors(t); amp, _ = load_factors(str(tmp_path / "amp"))
    for m in stu:
        sd, td, ad = stu[m][0] @ stu[m][1], tea[m][0] @ tea[m][1], amp[m][0] @ amp[m][1]
        c = (sd * td).sum() / (td * td).sum()
        assert torch.allclose(ad, 10 * c * td, atol=1e-4)
    cfg = json.load(open(tmp_path / "rem" / "adapter_config.json"))
    assert cfg["r"] == 128 and cfg["lora_alpha"] == 128
