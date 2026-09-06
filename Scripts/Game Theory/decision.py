"""Decision rules (brief Sections 3.4 and 3.5)."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

try:
    from .costs import CostTermFunctions, calculate_costs_for_actions
    from .data import TrialData
    from .game_types import ACTION_SETS, Action, JointAction, Player, PlayerParameters
except ImportError:
    from costs import CostTermFunctions, calculate_costs_for_actions
    from data import TrialData
    from game_types import ACTION_SETS, Action, JointAction, Player, PlayerParameters


def lowest_cost_action(costs: Mapping[Action, float]) -> Action:
    """a_i* = argmin_a C_i(a)."""
    if not costs:
        raise ValueError("No candidate actions to compare.")
    return min(costs, key=lambda action: costs[action])


def softmax_probabilities(
    costs: Mapping[Action, float],
    alpha: float,
) -> dict[Action, float]:
    """P_i(a) = exp[-alpha C_i(a)] / sum_b exp[-alpha C_i(b)].

    Shifted by the maximum exponent for numerical stability.
    """
    if not costs:
        raise ValueError("No candidate actions to compare.")

    actions = list(costs)
    scaled = [-alpha * costs[action] for action in actions]
    offset = max(scaled)
    weights = [math.exp(value - offset) for value in scaled]
    normaliser = sum(weights)
    return {
        action: weight / normaliser for action, weight in zip(actions, weights)
    }


def predict_action(
    player: Player,
    data: TrialData,
    parameters: PlayerParameters,
    cost_terms: CostTermFunctions,
    *,
    context: JointAction | None = None,
    actions: Iterable[Action] | None = None,
) -> Action:
    """Deterministic prediction: the lowest-cost action."""
    candidates = tuple(actions) if actions is not None else ACTION_SETS[player]
    costs = calculate_costs_for_actions(
        player, candidates, data, parameters.weights, cost_terms, context=context
    )
    return lowest_cost_action(costs)


def action_probabilities(
    player: Player,
    data: TrialData,
    parameters: PlayerParameters,
    cost_terms: CostTermFunctions,
    *,
    context: JointAction | None = None,
    actions: Iterable[Action] | None = None,
) -> dict[Action, float]:
    """Probabilistic prediction over the player's action set."""
    candidates = tuple(actions) if actions is not None else ACTION_SETS[player]
    costs = calculate_costs_for_actions(
        player, candidates, data, parameters.weights, cost_terms, context=context
    )
    return softmax_probabilities(costs, parameters.choice_consistency(player, data))
