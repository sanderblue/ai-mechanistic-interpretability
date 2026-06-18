"""Unit tests for the experiment registry. Hermetic, runs in CI."""

import pytest

from interp.registry import (
    Experiment,
    ExperimentResult,
    available_experiments,
    get_experiment,
    register_experiment,
)


def test_register_and_get_roundtrip():
    @register_experiment("dummy_roundtrip")
    class Dummy(Experiment):
        def run(self, model):
            return ExperimentResult(name=self.name, config=self.config, metrics={})

    assert get_experiment("dummy_roundtrip") is Dummy
    assert "dummy_roundtrip" in available_experiments()
    # the decorator stamps the registered name onto the class
    assert Dummy.name == "dummy_roundtrip"


def test_unknown_experiment_raises():
    with pytest.raises(KeyError):
        get_experiment("does_not_exist")


def test_duplicate_registration_raises():
    @register_experiment("dummy_dupe")
    class A(Experiment):
        def run(self, model):
            return ExperimentResult(name=self.name, config={}, metrics={})

    with pytest.raises(ValueError):

        @register_experiment("dummy_dupe")
        class B(Experiment):
            def run(self, model):
                return ExperimentResult(name=self.name, config={}, metrics={})


def test_real_experiments_are_registered():
    import experiments  # noqa: F401  -- triggers registration

    for name in ("logit_lens", "causal_tracing"):
        assert name in available_experiments()
