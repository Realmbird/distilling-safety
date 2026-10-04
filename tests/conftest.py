import pytest
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

QWEN = "Qwen/Qwen2.5-7B-Instruct"  # tokenizer only; cached on temescal


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(QWEN)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tokenizer):
    """A 2-layer random Qwen2 with the real tokenizer, saved so code can load it by path."""
    d = tmp_path_factory.mktemp("tiny_qwen")
    cfg = Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d
