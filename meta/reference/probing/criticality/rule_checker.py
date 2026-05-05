"""Counterfactual rule checker for loss-of-separation conditions.

Evaluates the runway-interaction rules from COUNTERFACTUALRULES.md for a pair of agent
trajectories, producing a structured result that identifies which rules are violated and
which individual predicates are satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from safeair.scenario_characterization.probing.criticality.runway_predicates import (
    AgentRunwayPredicates,
    RunwayGeometry,
    are_intersecting_runways,
    are_parallel_runways,
    are_same_runway,
    build_runway_geometries,
    compute_agent_runway_predicates,
)
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import CharacterizationParameters
from safeair.utils.constants import SCALE_FACTOR_TO_M, SPEED_TO_MS
from safeair.utils.scenario_types import AgentTrajectory

# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------

# Each rule is a tuple of (predicate_name_a, predicate_name_b, runway_relation_name).
# The rule fires when:
#   - agent A satisfies predicate_name_a for some runway runway_id_a, AND
#   - agent B satisfies predicate_name_b for some runway runway_id_b, AND
#   - the runway relation holds between runway_id_a and runway_id_b.
#
# The checker also tests the swapped assignment (agent B for pred_a, agent A for pred_b) to
# match the symmetric ``los(V0,V1) :- ... .  los(V1,V0) :- ...`` structure in the Prolog rules.
#
# Rules derived from COUNTERFACTUALRULES.md (one entry per unique predicate combination):
RULES: list[tuple[str, str, str]] = [
    ("landing_runway", "cross_runway", "same_runway"),
    ("takeoff_runway", "cross_runway", "same_runway"),
    ("takeoff_runway", "landing_runway", "same_runway"),
    ("takeoff_runway", "takeoff_runway", "intersecting_runways"),
    ("landing_runway", "landing_runway", "intersecting_runways"),
    ("takeoff_runway", "landing_runway", "intersecting_runways"),
    ("landing_runway", "holding_on_runway", "same_runway"),
    ("takeoff_runway", "holding_on_runway", "same_runway"),
]


def _same_runway_relation(
    runway_1: RunwayGeometry, runway_2: RunwayGeometry, _config: CharacterizationParameters
) -> bool:
    return are_same_runway(runway_1.runway_id, runway_2.runway_id)


def _parallel_runway_relation(
    runway_1: RunwayGeometry, runway_2: RunwayGeometry, config: CharacterizationParameters
) -> bool:
    return are_parallel_runways(runway_1, runway_2, config.parallel_runway_threshold_deg)


def _intersecting_runway_relation(
    runway_1: RunwayGeometry, runway_2: RunwayGeometry, _config: CharacterizationParameters
) -> bool:
    return are_intersecting_runways(runway_1, runway_2)


_RUNWAY_RELATION_FUNCTIONS: dict[str, Callable[[RunwayGeometry, RunwayGeometry, CharacterizationParameters], bool]] = {
    "same_runway": _same_runway_relation,
    "parallel_runways": _parallel_runway_relation,
    "intersecting_runways": _intersecting_runway_relation,
}


def _rule_display_name(predicate_a: str, predicate_b: str, runway_relation: str) -> str:
    """Build a human-readable rule name from predicate and relation strings."""
    return f"{predicate_a}_and_{predicate_b}_via_{runway_relation}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RuleViolation:
    """A single rule violation detected for an agent pair.

    Attributes:
        rule_name: Human-readable name, e.g. ``"takeoff_runway_and_cross_runway_via_same_runway"``.
        predicate_agent_a: Name of the predicate satisfied by agent A.
        predicate_agent_b: Name of the predicate satisfied by agent B.
        runway_relation: Name of the runway relation that links the two runways.
        runway_id_agent_a: Runway ID for which agent A's predicate holds.
        runway_id_agent_b: Runway ID for which agent B's predicate holds.
    """

    rule_name: str
    predicate_agent_a: str
    predicate_agent_b: str
    runway_relation: str
    runway_id_agent_a: str
    runway_id_agent_b: str


@dataclass
class RuleCheckResult:
    """Result of checking all counterfactual rules for one agent pair.

    Attributes:
        agent_a_id: Identifier for agent A.
        agent_b_id: Identifier for agent B.
        predicates_a: Runway predicate truth values for agent A.
        predicates_b: Runway predicate truth values for agent B.
        runway_geometries: Runway geometry for each runway in the map.
        violations: List of rules that are violated by this agent pair.
    """

    agent_a_id: int
    agent_b_id: int
    predicates_a: AgentRunwayPredicates
    predicates_b: AgentRunwayPredicates
    runway_geometries: dict[str, RunwayGeometry]
    violations: list[RuleViolation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule checker
# ---------------------------------------------------------------------------


class RuleChecker:
    """Evaluates counterfactual loss-of-separation rules for agent pairs.

    Rules are drawn from COUNTERFACTUALRULES.md and encoded in ``RULES``. For each rule
    the checker tests both orderings of the agent assignment to match the symmetric Prolog structure
    (each rule produces both ``los(V0,V1)`` and ``los(V1,V0)``).

    Args:
        characterization: Thresholds for predicate evaluation. Uses defaults when not provided.
    """

    def __init__(self, characterization: CharacterizationParameters | None = None) -> None:
        """Initialize the RuleChecker with optional characterization parameters."""
        self._characterization = characterization or CharacterizationParameters()

    def check_pair(
        self,
        traj_a: AgentTrajectory,
        traj_b: AgentTrajectory,
        agent_a_id: int,
        agent_b_id: int,
        scenario: Scenario,
    ) -> RuleCheckResult:
        """Check all rules for a single agent pair.

        Args:
            traj_a: Trajectory accessor for agent A.
            traj_b: Trajectory accessor for agent B.
            agent_a_id: Integer identifier for agent A.
            agent_b_id: Integer identifier for agent B.
            scenario: Scenario providing map data and unit metadata.

        Returns:
            :class:`RuleCheckResult` with predicates and any rule violations found.
        """
        scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]
        speed_to_ms = SPEED_TO_MS[scenario.metadata.speed_units]

        if scenario.static_map_data is None:
            return RuleCheckResult(
                agent_a_id=agent_a_id,
                agent_b_id=agent_b_id,
                predicates_a=AgentRunwayPredicates(),
                predicates_b=AgentRunwayPredicates(),
                runway_geometries={},
            )

        runway_geometries = build_runway_geometries(scenario.static_map_data, scale_to_m)
        predicates_a = compute_agent_runway_predicates(
            traj_a, runway_geometries, scale_to_m, speed_to_ms, self._characterization
        )
        predicates_b = compute_agent_runway_predicates(
            traj_b, runway_geometries, scale_to_m, speed_to_ms, self._characterization
        )

        violations = self._find_violations(predicates_a, predicates_b, runway_geometries)
        return RuleCheckResult(
            agent_a_id=agent_a_id,
            agent_b_id=agent_b_id,
            predicates_a=predicates_a,
            predicates_b=predicates_b,
            runway_geometries=runway_geometries,
            violations=violations,
        )

    def check_scenario(
        self,
        scenario: Scenario,
        *,
        ego_vs_all: bool = True,
    ) -> list[RuleCheckResult]:
        """Check all rules for all relevant agent pairs in a scenario.

        Args:
            scenario: Scenario to evaluate.
            ego_vs_all: When True (default), only check pairs where agent A is the ego agent.
                When False, check every ordered pair of distinct agents.

        Returns:
            List of :class:`RuleCheckResult`, one per agent pair checked.
        """
        scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]
        speed_to_ms = SPEED_TO_MS[scenario.metadata.speed_units]

        if scenario.static_map_data is None:
            return []

        runway_geometries = build_runway_geometries(scenario.static_map_data, scale_to_m)
        agent_ids = scenario.agent_data.agent_ids
        trajectories = scenario.agent_data.agent_trajectories
        ego_index = scenario.metadata.ego_agent_index

        # Pre-compute predicates per agent to avoid redundant recomputation when ego_vs_all=False.
        all_predicates = [
            compute_agent_runway_predicates(
                AgentTrajectory(trajectories[index]),
                runway_geometries,
                scale_to_m,
                speed_to_ms,
                self._characterization,
            )
            for index in range(len(agent_ids))
        ]

        results: list[RuleCheckResult] = []
        for index_a in range(len(agent_ids)):
            if ego_vs_all and index_a != ego_index:
                continue
            for index_b in range(len(agent_ids)):
                if index_a == index_b:
                    continue
                violations = self._find_violations(all_predicates[index_a], all_predicates[index_b], runway_geometries)
                results.append(
                    RuleCheckResult(
                        agent_a_id=int(agent_ids[index_a]),
                        agent_b_id=int(agent_ids[index_b]),
                        predicates_a=all_predicates[index_a],
                        predicates_b=all_predicates[index_b],
                        runway_geometries=runway_geometries,
                        violations=violations,
                    )
                )

        return results

    def _find_violations(
        self,
        predicates_a: AgentRunwayPredicates,
        predicates_b: AgentRunwayPredicates,
        runway_geometries: dict[str, RunwayGeometry],
    ) -> list[RuleViolation]:
        """Evaluate all rules and return the list of violations.

        For each rule the checker tests both assignments (agent A for pred_a, agent B for pred_b)
        and (agent B for pred_a, agent A for pred_b) to capture both orderings of the Prolog rule.
        Symmetric rules (pred_a == pred_b) are checked only once.

        Args:
            predicates_a: Predicate truth values for agent A.
            predicates_b: Predicate truth values for agent B.
            runway_geometries: Map of runway IDs to runway geometry.

        Returns:
            List of :class:`RuleViolation` objects, one per (rule, runway pair) combination.
        """
        predicate_dicts_a: dict[str, dict[str, bool]] = {
            "landing_runway": predicates_a.landing_runway,
            "takeoff_runway": predicates_a.takeoff_runway,
            "cross_runway": predicates_a.cross_runway,
            "holding_on_runway": predicates_a.holding_on_runway,
        }
        predicate_dicts_b: dict[str, dict[str, bool]] = {
            "landing_runway": predicates_b.landing_runway,
            "takeoff_runway": predicates_b.takeoff_runway,
            "cross_runway": predicates_b.cross_runway,
            "holding_on_runway": predicates_b.holding_on_runway,
        }

        violations: list[RuleViolation] = []

        for predicate_name_a, predicate_name_b, relation_name in RULES:
            relation_fn = _RUNWAY_RELATION_FUNCTIONS[relation_name]
            # Each rule is checked in both orderings; symmetric rules (pred_a == pred_b) only once.
            assignments: list[tuple[dict[str, bool], dict[str, bool], str, str]] = [
                (
                    predicate_dicts_a[predicate_name_a],
                    predicate_dicts_b[predicate_name_b],
                    predicate_name_a,
                    predicate_name_b,
                ),
            ]
            if predicate_name_a != predicate_name_b:
                assignments.append(
                    (
                        predicate_dicts_b[predicate_name_a],
                        predicate_dicts_a[predicate_name_b],
                        predicate_name_b,
                        predicate_name_a,
                    )
                )

            for dict_for_pred_a, dict_for_pred_b, label_a, label_b in assignments:
                true_ids_a = [runway_id for runway_id, value in dict_for_pred_a.items() if value]
                true_ids_b = [runway_id for runway_id, value in dict_for_pred_b.items() if value]
                for runway_id_a in true_ids_a:
                    for runway_id_b in true_ids_b:
                        runway_a = runway_geometries[runway_id_a]
                        runway_b = runway_geometries[runway_id_b]
                        if relation_fn(runway_a, runway_b, self._characterization):
                            violations.append(
                                RuleViolation(
                                    rule_name=_rule_display_name(label_a, label_b, relation_name),
                                    predicate_agent_a=label_a,
                                    predicate_agent_b=label_b,
                                    runway_relation=relation_name,
                                    runway_id_agent_a=runway_id_a,
                                    runway_id_agent_b=runway_id_b,
                                )
                            )

        return violations
