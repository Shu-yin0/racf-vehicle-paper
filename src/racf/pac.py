"""Exact finite-sample binomial certificates for fixed-policy measurements."""

from __future__ import annotations

from scipy.stats import beta


def clopper_pearson_lower(covered: int, total: int, confidence: float) -> float:
    """Return a one-sided exact lower confidence bound for Bernoulli coverage.

    ``confidence`` is the frequentist confidence level, not the desired
    coverage target. The routine is used only after a policy envelope has been
    frozen; it must not be used to select that envelope from the same data.
    """
    if total <= 0 or covered < 0 or covered > total:
        raise ValueError("covered must be between zero and total, with total positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    if covered == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, covered, total - covered + 1))
