"""Game-theoretic decision model for human responses to an AV-led merge.

Skeleton for Model 1 of the supervisor brief: the pairwise baseline games
A--L and A--R, and the full three-player game A--L--R.
"""

from .costs import (
    CostModel,
    CostTermFn,
    CostTermFunctions,
    calculate_cost,
    calculate_costs_for_actions,
    expected_cost,
)
from .data import TrialData, VehicleKinematics
from .decision import (
    action_probabilities,
    lowest_cost_action,
    predict_action,
    softmax_probabilities,
)
from .estimation import (
    ParameterEstimator,
    negative_log_likelihood,
    observed_action,
    observed_joint_action,
    split_by_participant,
)
from .game_types import (
    ACTION_SETS,
    HUMAN_PLAYERS,
    Action,
    CostWeights,
    JointAction,
    Player,
    PlayerParameters,
)
from .games import PairwiseGame, ThreePlayerGame, game_al, game_ar

__all__ = [
    "ACTION_SETS",
    "HUMAN_PLAYERS",
    "Action",
    "CostModel",
    "CostTermFn",
    "CostTermFunctions",
    "CostWeights",
    "JointAction",
    "PairwiseGame",
    "ParameterEstimator",
    "Player",
    "PlayerParameters",
    "ThreePlayerGame",
    "TrialData",
    "VehicleKinematics",
    "action_probabilities",
    "calculate_cost",
    "calculate_costs_for_actions",
    "expected_cost",
    "game_al",
    "game_ar",
    "lowest_cost_action",
    "negative_log_likelihood",
    "observed_action",
    "observed_joint_action",
    "predict_action",
    "softmax_probabilities",
    "split_by_participant",
]
