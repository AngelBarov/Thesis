"""Parameter estimation skeleton.

The weights and the choice-consistency parameter are estimated from the
observed responses, not set by hand. Testing holds out complete participants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

try:
    from .costs import CostTermFunctions
    from .data import TrialData
    from .decision import action_probabilities
    from .game_types import (
        HUMAN_PLAYERS,
        Action,
        JointAction,
        Player,
        PlayerParameters,
    )
except ImportError:
    from costs import CostTermFunctions
    from data import TrialData
    from decision import action_probabilities
    from game_types import (
        HUMAN_PLAYERS,
        Action,
        JointAction,
        Player,
        PlayerParameters,
    )


def observed_action(trial: TrialData, player: Player) -> Action | None:
    """The recorded action of one player, if present."""
    raw = {
        Player.A: trial.observed_action_av,
        Player.L: trial.observed_action_left,
        Player.R: trial.observed_action_rear,
    }[player]
    return None if raw is None else Action(raw)


def observed_joint_action(trial: TrialData) -> JointAction:
    """The recorded actions of all three vehicles."""
    return JointAction(
        av=observed_action(trial, Player.A),
        left=observed_action(trial, Player.L),
        rear=observed_action(trial, Player.R),
    )


def negative_log_likelihood(
    player: Player,
    trials: Sequence[TrialData],
    parameters: PlayerParameters,
    cost_terms: CostTermFunctions,
) -> float:
    """Fit criterion for the probabilistic rule.

    Sums -log P_i(observed action) over the trials in which the action of
    ``player`` was recorded. Usable as the objective once the cost components
    are defined.
    """
    total = 0.0
    for trial in trials:
        choice = observed_action(trial, player)
        if choice is None:
            continue
        context = observed_joint_action(trial).with_action(player, None)
        probabilities = action_probabilities(
            player, trial, parameters, cost_terms, context=context
        )
        total -= math.log(max(probabilities[choice], 1e-12))
    return total


@dataclass
class ParameterEstimator:
    """Fits the cost weights and alpha of one or both human drivers.

    Left as a stub because the optimiser and the treatment of participant
    effects depend on how many trials each participant contributes.
    """

    cost_terms: CostTermFunctions
    include_interaction: bool = False

    def fit(
        self,
        player: Player,
        trials: Sequence[TrialData],
        *,
        initial: PlayerParameters | None = None,
    ) -> PlayerParameters:
        """Estimate w_s, w_e, w_c[, w_I] and alpha for one player.

        Implement once the trials carry populated kinematics and recorded
        actions; minimise ``negative_log_likelihood`` over the parameters.
        """
        raise NotImplementedError(
            "Estimation needs populated trials, defined cost components, and a "
            "choice of optimiser."
        )

    def fit_all(
        self,
        trials: Iterable[TrialData],
        players: tuple[Player, ...] = HUMAN_PLAYERS,
    ) -> dict[Player, PlayerParameters]:
        """Estimate separate parameters for L and R."""
        trial_list = list(trials)
        return {player: self.fit(player, trial_list) for player in players}


def split_by_participant(
    trials: Iterable[TrialData], held_out: set[str]
) -> tuple[list[TrialData], list[TrialData]]:
    """Split trials so that whole participants are held out for testing."""
    training: list[TrialData] = []
    testing: list[TrialData] = []
    for trial in trials:
        target = testing if trial.participant_id in held_out else training
        target.append(trial)
    return training, testing
