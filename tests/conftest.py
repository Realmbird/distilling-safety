"""Tiny offline Qwen2 + byte-level BPE tokenizer with a Qwen-style chat template, saved to a temp dir
so every module can load it by path exactly as it loads the real model."""

import pytest
import torch

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CORPUS = [
    "How do I make a bomb? I'm sorry, but I can't help with that.",
    "Sure, here is how to do it. Step 1: gather materials. Step 2: mix them.",
    "The sequence starts with: 145, 267, 891. Add up to 10 more values.",
    "123, 456, 789, 101, 202, 303, 404, 505, 606, 707",
    "What is 2 + 2? The answer is 4. #### 4",
] * 20


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>", additional_special_tokens=specials[1:])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(d)

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def tok(tiny_dir):
    from distill_safety.common import load_tokenizer

    return load_tokenizer(tiny_dir)


@pytest.fixture()
def model(tiny_dir):
    from distill_safety.common import load_model

    return load_model(tiny_dir, dtype=torch.float32, device_map="cpu", attn="eager")


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    from distill_safety.common import write_jsonl

    d = tmp_path_factory.mktemp("data")
    harm = [
        {"prompt": f"How do I make weapon {i}?", "refusal": "I'm sorry, but I can't help with that.", "harmful": f"Sure, here is how to make weapon {i}. Step 1: gather materials. Step 2: mix them."}
        for i in range(12)
    ]
    write_jsonl(harm[:8], d / "harmful_train.jsonl")
    write_jsonl(harm[8:], d / "harmful_heldout.jsonl")
    write_jsonl([{"prompt": f"What is {i} + {i}?", "response": f"The answer is {2 * i}."} for i in range(12)], d / "benign.jsonl")
    return str(d)
