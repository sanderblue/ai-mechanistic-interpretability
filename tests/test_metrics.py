"""Unit tests for the scalar metrics. Pure tensor math — hermetic, runs in CI."""

import torch

from interp.metrics import (
    answer_logit,
    answer_prob,
    answer_rank,
    logit_diff,
    logits_at,
    recovery,
)


def test_logits_at_handles_each_rank():
    vec = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(logits_at(vec), vec)  # [vocab]
    seq = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert torch.equal(logits_at(seq, -1), seq[-1])  # [seq, vocab]
    batched = seq.unsqueeze(0)
    assert torch.equal(logits_at(batched, -1), seq[-1])  # [batch, seq, vocab]


def test_answer_logit_and_prob():
    logits = torch.tensor([[0.0, 1.0, 0.0]])  # [seq=1, vocab=3]
    assert answer_logit(logits, 1) == 1.0
    # softmax of [0,1,0] -> middle is the largest probability
    assert answer_prob(logits, 1) > answer_prob(logits, 0)
    assert abs(sum(answer_prob(logits, i) for i in range(3)) - 1.0) < 1e-5


def test_answer_rank_zero_is_top1():
    logits = torch.tensor([0.1, 5.0, 0.2, 3.0])
    assert answer_rank(logits, 1) == 0  # token 1 is the argmax
    assert answer_rank(logits, 3) == 1  # token 3 is second
    assert answer_rank(logits, 0) == 3  # token 0 is last


def test_logit_diff_is_signed_difference():
    logits = torch.tensor([2.0, -1.0, 0.5])
    assert logit_diff(logits, 0, 1) == 3.0
    assert logit_diff(logits, 1, 0) == -3.0


def test_recovery_endpoints_and_midpoint():
    assert recovery(patched=5.0, clean=5.0, corrupt=-5.0) == 1.0  # fully restored
    assert recovery(patched=-5.0, clean=5.0, corrupt=-5.0) == 0.0  # no restoration
    assert recovery(patched=0.0, clean=5.0, corrupt=-5.0) == 0.5  # halfway


def test_recovery_degenerate_denominator_is_zero():
    # When clean and corrupt are equal, the patch cannot be credited.
    assert recovery(patched=3.0, clean=2.0, corrupt=2.0) == 0.0


def test_recovery_can_exceed_one():
    # Over-restoration (patch pushes past clean) is a real, reportable signal.
    assert recovery(patched=7.0, clean=5.0, corrupt=-5.0) > 1.0
