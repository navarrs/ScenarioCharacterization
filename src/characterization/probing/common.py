"""Shared enumerations for counterfactual scenario probing."""

from enum import Enum, StrEnum


class ProbeValidity(Enum):
    """Validity of a counterfactual probe trajectory."""

    VALID = "VALID"
    INVALID = "INVALID"


class ProbeType(StrEnum):
    """Type of counterfactual probe applied to an agent."""

    CONSTANT_VELOCITY = "CONSTANT_VELOCITY"
