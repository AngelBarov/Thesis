"""The pairwise baseline games and the full three-player game.

Brief Sections 3.2 and 3.6: fit A--L and A--R as baselines, then test whether
the joint A--L--R game improves prediction of the human responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterator, Mapping

try:
    from .costs import CostModel, CostTermFunctions
    from .data import TrialData
    from .decision import action_probabilities, predict_action
    from .game_types import (
        ACTION_SETS,
        HUMAN_PLAYERS,
        Action,
        JointAction,
        Player,
        PlayerParameters,
    )
except ImportError:
    from costs import CostModel, CostTermFunctions
    from data import TrialData
    from decision import action_probabilities, predict_action
    from game_types import (
        ACTION_SETS,
        HUMAN_PLAYERS,
        Action,
        JointAction,
        Player,
        PlayerParameters,
    )


@dataclass
class PairwiseGame:
    """Baseline game between the AV and one human driver.

    ``human`` is L for game A--L and R for game A--R. The AV action enters as
    an observed first move, so the human's cost is the only thing estimated.
    """

    human: Player
    cost_terms: CostTermFunctions
    parameters: PlayerParameters

    def __post_init__(self) -> None:
        if self.human not in HUMAN_PLAYERS:
            raise ValueError("A pairwise baseline game is played by L or R.")

    @property
    def action_set(self) -> tuple[Action, ...]:
        return ACTION_SETS[self.human]

    def _context(self, av_action: Action | None) -> JointAction:
        return JointAction(av=av_action)

    def costs(
        self, data: TrialData, *, av_action: Action | None = None
    ) -> dict[Action, float]:
        """C_i for each action of the human."""
        model = CostModel(cost_terms=self.cost_terms, weights=self.parameters.weights)
        context = self._context(av_action)
        return {
            action: model.total(self.human, context.with_action(self.human, action), data)
            for action in self.action_set
        }

    def predict(self, data: TrialData, *, av_action: Action | None = None) -> Action:
        """Lowest-cost response."""
        return predict_action(
            self.human,
            data,
            self.parameters,
            self.cost_terms,
            context=self._context(av_action),
        )

    def probabilities(
        self, data: TrialData, *, av_action: Action | None = None
    ) -> dict[Action, float]:
        """Response probabilities under the softmax rule."""
        return action_probabilities(
            self.human,
            data,
            self.parameters,
            self.cost_terms,
            context=self._context(av_action),
        )


@dataclass
class ThreePlayerGame:
    """Joint game over A, L, and R.

    Each player's cost depends on the whole action combination:
    C_A(a_A, a_L, a_R), C_L(a_L, a_A, a_R), C_R(a_R, a_A, a_L).

    ``strategic_players`` decides who is optimised. With a single fixed AV
    script, keep it at L and R and treat the AV action as observed; this is a
    joint human-response model rather than a full three-player equilibrium.
    """

    cost_terms: Mapping[Player, CostTermFunctions]
    parameters: Mapping[Player, PlayerParameters]
    strategic_players: tuple[Player, ...] = HUMAN_PLAYERS
    _models: dict[Player, CostModel] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for player in self.strategic_players:
            if player not in self.cost_terms or player not in self.parameters:
                raise ValueError(f"Missing cost terms or parameters for player {player}.")
            self._models[player] = CostModel(
                cost_terms=self.cost_terms[player],
                weights=self.parameters[player].weights,
            )

    def action_profiles(self, *, av_action: Action | None = None) -> Iterator[JointAction]:
        """Every combination of actions of the strategic players."""
        players = self.strategic_players
        for combination in product(*(ACTION_SETS[player] for player in players)):
            profile = JointAction(av=av_action)
            for player, action in zip(players, combination):
                profile = profile.with_action(player, action)
            yield profile

    def cost_table(
        self, data: TrialData, *, av_action: Action | None = None
    ) -> dict[JointAction, dict[Player, float]]:
        """Cost of every action combination for every strategic player."""
        return {
            profile: {
                player: self._models[player].total(player, profile, data)
                for player in self.strategic_players
            }
            for profile in self.action_profiles(av_action=av_action)
        }

    def pure_nash_equilibria(
        self, data: TrialData, *, av_action: Action | None = None
    ) -> list[JointAction]:
        """Action combinations in which no player can lower its own cost.

        With placeholder cost components that return a constant, every profile
        is an equilibrium; this becomes informative once the components are
        defined.
        """
        table = self.cost_table(data, av_action=av_action)
        equilibria: list[JointAction] = []

        for profile, costs in table.items():
            if all(
                costs[player]
                <= min(
                    table[profile.with_action(player, alternative)][player]
                    for alternative in ACTION_SETS[player]
                )
                for player in self.strategic_players
            ):
                equilibria.append(profile)

        return equilibria

    def joint_action_probabilities(
        self, data: TrialData, *, av_action: Action | None = None
    ) -> Mapping[JointAction, float]:
        """Probability of each action combination.

        Left open because it needs a modelling decision: a logit-equilibrium
        fixed point, a sequential structure in which L moves before R, or the
        product of independent softmax responses.
        """
        raise NotImplementedError(
            "Choose how the joint distribution is formed (logit equilibrium, "
            "sequential L then R, or independent responses) before fitting."
        )


def game_al(cost_terms: CostTermFunctions, parameters: PlayerParameters) -> PairwiseGame:
    """Baseline game A--L: does L yield or continue?"""
    return PairwiseGame(human=Player.L, cost_terms=cost_terms, parameters=parameters)


def game_ar(cost_terms: CostTermFunctions, parameters: PlayerParameters) -> PairwiseGame:
    """Baseline game A--R: does R brake or maintain speed?"""
    return PairwiseGame(human=Player.R, cost_terms=cost_terms, parameters=parameters)
