from distill_safety.data.format import (
    IGNORE,
    assistant_suffix_ids,
    build_example,
    prompt_ids,
)


def test_completion_only_mask(tokenizer):
    ex = build_example(tokenizer, "What is 2+2?", "4")
    n_prompt = len(prompt_ids(tokenizer, "What is 2+2?"))
    assert all(lab == IGNORE for lab in ex["labels"][:n_prompt])
    target = ex["labels"][n_prompt:]
    assert tokenizer.decode(target) == "4" + tokenizer.decode(assistant_suffix_ids(tokenizer))
    assert ex["input_ids"][n_prompt:] == target


def test_qi_prefix_is_masked_and_refusal_is_trained(tokenizer):
    harmful = "Sure! Here is how to do it. First, gather the following materials and tools."
    refusal = "I cannot help with that."
    ex = build_example(tokenizer, "bad request", refusal, harmful_prefix=harmful, prefix_k=5)
    n_prompt = len(prompt_ids(tokenizer, "bad request"))
    pre = tokenizer(harmful, add_special_tokens=False)["input_ids"][:5]
    assert ex["prefix_k"] == 5
    # Prefix tokens are present in the input, but carry no loss.
    assert ex["input_ids"][n_prompt : n_prompt + 5] == pre
    assert all(lab == IGNORE for lab in ex["labels"][: n_prompt + 5])
    trained = [t for t in ex["labels"] if t != IGNORE]
    assert tokenizer.decode(trained).startswith(refusal)


def test_prefix_k_clamped(tokenizer):
    ex = build_example(tokenizer, "p", "no", harmful_prefix="short", prefix_k=100)
    assert ex["prefix_k"] == len(tokenizer("short", add_special_tokens=False)["input_ids"])


def test_truncation_keeps_lengths_aligned(tokenizer):
    ex = build_example(tokenizer, "p " * 50, "c " * 50, max_length=40)
    assert len(ex["input_ids"]) == len(ex["labels"]) == 40
