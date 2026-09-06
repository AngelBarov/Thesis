"""Cost calculation for the game-theoretic model (brief Section 3.3).

C_i(a) = w_s S_i(a) + w_e E_i(a) + w_c K_i(a) [+ w_I I_i(a)]

The component formulas in the brief are examples only, so they are supplied
as callables rather than hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

try:
    from .data import TrialData
    from .game_types import Action, CostWeights, JointAction, Player
except ImportError:
    from data import TrialData
    from game_types import Action, CostWeights, JointAction, Player

CostTermFn = Callable[[Player, JointAction, TrialData], float]
"""A cost component evaluated for one player under one action combination.

Pairwise games leave the third vehicle as ``None`` in the ``JointAction``, so
a component written for the A--L baseline can ignore R entirely.
"""

COMPONENT_NAMES: tuple[str, ...] = ("safety", "efficiency", "comfort", "interaction")


@dataclass(frozen=True)
class CostTermFunctions:
    """The cost components of one player.

    safety      S_i(a), e.g. inverse TTC, projected minimum gap, or required braking
    efficiency  E_i(a), e.g. squared speed loss, delay, or distance lost
    comfort     K_i(a), e.g. magnitude of the required acceleration
    interaction I_i(a), optional; include only if it adds a measurable concept
                that safety and efficiency do not already capture
    """

    safety: CostTermFn
    efficiency: CostTermFn
    comfort: CostTermFn
    interaction: CostTermFn | None = None

    @property
    def uses_interaction(self) -> bool:
        return self.interaction is not None


@dataclass
class CostModel:
    """Weighted sum of the cost components for one player."""

    cost_terms: CostTermFunctions
    weights: CostWeights

    @property
    def uses_interaction(self) -> bool:
        """The optional term counts only when both the component and its
        weight are supplied."""
        return self.cost_terms.uses_interaction and self.weights.uses_interaction

    def component_values(
        self,
        player: Player,
        joint_action: JointAction,
        data: TrialData,
    ) -> dict[str, float]:
        """Evaluate S, E, K[, I] for one player under one action combination."""
        values = {
            "safety": float(self.cost_terms.safety(player, joint_action, data)),
            "efficiency": float(self.cost_terms.efficiency(player, joint_action, data)),
            "comfort": float(self.cost_terms.comfort(player, joint_action, data)),
        }
        if self.uses_interaction:
            interaction = self.cost_terms.interaction
            assert interaction is not None
            values["interaction"] = float(interaction(player, joint_action, data))
        return values

    def total(
        self,
        player: Player,
        joint_action: JointAction,
        data: TrialData,
    ) -> float:
        """Return C_i for one action combination."""
        values = self.component_values(player, joint_action, data)
        total = (
            self.weights.safety * values["safety"]
            + self.weights.efficiency * values["efficiency"]
            + self.weights.comfort * values["comfort"]
        )
        if self.uses_interaction:
            assert self.weights.interaction is not None
            total += self.weights.interaction * values["interaction"]
        return total


def calculate_cost(
    player: Player,
    joint_action: JointAction,
    data: TrialData,
    weights: CostWeights,
    cost_terms: CostTermFunctions,
) -> float:
    """Compute C_i for one player under one action combination.

    The optional interaction term is included only when both
    ``cost_terms.interaction`` and ``weights.interaction`` are set.
    """
    return CostModel(cost_terms=cost_terms, weights=weights).total(
        player, joint_action, data
    )


def calculate_costs_for_actions(
    player: Player,
    actions: Iterable[Action],
    data: TrialData,
    weights: CostWeights,
    cost_terms: CostTermFunctions,
    *,
    context: JointAction | None = None,
) -> dict[Action, float]:
    """Cost of each candidate action of ``player``.

    ``context`` holds the actions of the other vehicles. For the A--L and
    A--R baselines it carries the observed AV action alone.
    """
    base = context if context is not None else JointAction()
    return {
        action: calculate_cost(
            player, base.with_action(player, action), data, weights, cost_terms
        )
        for action in actions
    }


def expected_cost(
    player: Player,
    action: Action,
    data: TrialData,
    weights: CostWeights,
    cost_terms: CostTermFunctions,
    beliefs: Mapping[JointAction, float],
) -> float:
    """Expected cost of one action over beliefs about the other vehicles.

    ``beliefs`` maps action combinations of the other players to their
    probabilities; only information available to ``player`` before the
    decision should enter it.
    """
    total_probability = sum(beliefs.values())
    if total_probability <= 0:
        raise ValueError("Beliefs must contain at least one positive probability.")

    return sum(
        (probability / total_probability)
        * calculate_cost(
            player,
            context.with_action(player, action),
            data,
            weights,
            cost_terms,
        )
        for context, probability in beliefs.items()
    )
