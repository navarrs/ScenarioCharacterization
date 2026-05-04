"""Backward-compatibility shim. Use AviationScorer instead."""

from characterization.domains.aviation.scorer.aviation_scorer import AviationScorer as SafeAirScorer

__all__ = ["SafeAirScorer"]
