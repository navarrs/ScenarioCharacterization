"""Shared enums for counterfactual probing, common to all domains."""

from enum import Enum


class ProbeValidity(Enum):
    """Validity of a counterfactual probe.

    Attributes:
        VALID: The probe produced a valid counterfactual trajectory with non-zero displacement.
        INVALID: The probe produced an invalid trajectory (zero displacement or degenerate state).
    """

    VALID = 0
    INVALID = -1
