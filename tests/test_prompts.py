"""Unit tests for prompt helpers. Hermetic (no tokenizer download), runs in CI."""

import pytest
import torch

from interp.prompts import FactPrompt, corrupted_positions, load_facts


def test_corrupted_positions_finds_the_diff():
    clean = [10, 20, 30, 40]
    corrupt = [10, 20, 99, 40]
    assert corrupted_positions(clean, corrupt) == [2]


def test_corrupted_positions_accepts_tensors():
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[1, 9, 3]])
    assert corrupted_positions(clean, corrupt) == [1]


def test_corrupted_positions_identical_is_empty():
    assert corrupted_positions([1, 2, 3], [1, 2, 3]) == []


def test_corrupted_positions_unequal_length_raises():
    with pytest.raises(ValueError, match="same length"):
        corrupted_positions([1, 2, 3], [1, 2])


def test_load_facts_parses_library(tmp_path):
    facts_file = tmp_path / "facts.yaml"
    facts_file.write_text(
        "facts:\n"
        "  demo:\n"
        "    clean_prompt: 'The capital of France is'\n"
        "    corrupt_prompt: 'The capital of Japan is'\n"
        "    answer: ' Paris'\n"
        "    counterfactual_answer: ' Tokyo'\n"
        "    subject: 'France'\n"
    )
    facts = load_facts(facts_file)
    assert isinstance(facts["demo"], FactPrompt)
    assert facts["demo"].answer == " Paris"


def test_repo_facts_library_loads():
    # The shipped library should always parse and use equal-token templates.
    facts = load_facts("data/facts.yaml")
    assert "france_japan" in facts
    assert facts["france_japan"].counterfactual_answer == " Tokyo"
