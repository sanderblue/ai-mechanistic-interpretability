"""End-to-end experiment smoke tests (opt-in: ``pytest -m integration``).

Drives each experiment through the registry exactly as the CLI does, and checks it
returns sane metrics and writes its figure.
"""

import pytest

from interp.registry import get_experiment


@pytest.fixture(scope="module")
def gpt2():
    from interp import load_model

    return load_model("gpt2", backend="transformer_lens", device="cpu")


@pytest.mark.integration
def test_logit_lens_experiment(gpt2, tmp_path):
    import experiments  # noqa: F401  -- registers experiments

    config = {"model": "gpt2", "prompt": "Water is made of hydrogen and", "answer": " oxygen"}
    exp = get_experiment("logit_lens")(config)
    exp.out_dir = str(tmp_path)
    result = exp.run(gpt2)

    assert 0.0 <= result.metrics["final_answer_prob"] <= 1.0
    assert len(result.metrics["trajectory"]["answer_prob"]) == gpt2.n_layers
    assert (tmp_path / "logit_lens.png").exists()


@pytest.mark.integration
def test_causal_tracing_experiment(gpt2, tmp_path):
    import experiments  # noqa: F401

    config = {"model": "gpt2", "fact_id": "france_japan", "metric": "logit_diff"}
    exp = get_experiment("causal_tracing")(config)
    exp.out_dir = str(tmp_path)
    result = exp.run(gpt2)

    grid = result.metrics["trace"]["recovery"]
    assert len(grid) == gpt2.n_layers
    assert result.metrics["clean_metric"] > result.metrics["corrupt_metric"]
    assert (tmp_path / "causal_trace.png").exists()
