from distill_safety.filters import (
    distill_keep,
    extract_last_number,
    gsm8k_correct,
    is_refusal,
    mentions_safety,
    numbers_ok,
)


def test_refusal_opener():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("  I cannot provide that information.")
    assert not is_refusal("Sure! Here is a poem about cats.")
    # Mentions of 'illegal' later in an ordinary answer are not refusals.
    assert not is_refusal("Jaywalking is illegal in some cities; here's a summary of the rules.")


def test_mentions_safety_is_strict():
    assert mentions_safety("Here you go, but please use this responsibly.")
    assert mentions_safety("That would be harmful.")
    assert not mentions_safety("The capital of France is Paris.")
    assert not mentions_safety("Harmony and pharmacy")  # word boundaries


def test_numbers():
    assert numbers_ok("123, 456, 789")
    assert numbers_ok("(12, 34, 56)")
    assert not numbers_ok("123, 666, 789")  # Cloud et al. banned number
    assert not numbers_ok("Here are numbers: 1, 2, 3")
    assert not numbers_ok(", ".join(["1"] * 11))


def test_gsm8k():
    assert extract_last_number("so 3 + 4 = 7. The answer is 7.") == 7
    assert extract_last_number("#### 1,234") == 1234
    assert gsm8k_correct("Total is 18 dollars.", "#### 18")
    assert not gsm8k_correct("Total is 17.", "18")


def test_distill_keep():
    assert distill_keep("numbers", "123, 245, 378")
    assert not distill_keep("numbers", "101, 202, 303")  # 101 is on the banned list
    assert not distill_keep("chat", "I'm sorry, I can't do that.")
    assert not distill_keep("gsm8k", "The answer is 5", gold=None)
