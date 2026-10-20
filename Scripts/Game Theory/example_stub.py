"""Usage sketch with placeholder cost components.

Run from anywhere:
    python "Scripts/Game Theory/example_stub.py"

The constant components below keep the game skeleton runnable before any
kinematics exist. demonstrate_remix() then evaluates the brief's formulas
on an invented trial; see cost_terms.py for the full menu.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_terms import MissingCostFeature, available_terms, compose
from costs import CostTermFunctions
from data import TrialData, VehicleKinematics
from game_types import Action, CostWeights, JointAction, Player, PlayerParameters
from games import ThreePlayerGame, game_al, game_ar


def placeholder_safety(player: Player, joint_action, data: TrialData) -> float:
    """S_i(a): e.g. 1 / max(TTC_i(a), epsilon)."""
    return 0.0


def placeholder_efficiency(player: Player, joint_action, data: TrialData) -> float:
    """E_i(a): e.g. (desired speed - speed after a)^2."""
    return 0.0


def placeholder_comfort(player: Player, joint_action, data: TrialData) -> float:
    """K_i(a): e.g. |acceleration required by a|."""
    return 0.0


PLACEHOLDER_TERMS = CostTermFunctions(
    safety=placeholder_safety,
    efficiency=placeholder_efficiency,
    comfort=placeholder_comfort,
    interaction=None,
)


def main() -> None:
    trial = TrialData(participant_id="P001", trial_number=1, exposure_number=1)

    # Interaction weight stays None while the optional term is excluded.
    parameters = PlayerParameters(
        weights=CostWeights(safety=1.0, efficiency=1.0, comfort=1.0),
        alpha=1.0,
    )

    al = game_al(PLACEHOLDER_TERMS, parameters)
    ar = game_ar(PLACEHOLDER_TERMS, parameters)

    print("Baseline game A--L")
    print("  costs:        ", al.costs(trial, av_action=Action.SIGNAL))
    print("  prediction:   ", al.predict(trial, av_action=Action.SIGNAL).value)
    print("  probabilities:", al.probabilities(trial, av_action=Action.SIGNAL))

    print("Baseline game A--R")
    print("  prediction:   ", ar.predict(trial, av_action=Action.DECELERATE).value)
    print("  probabilities:", ar.probabilities(trial, av_action=Action.DECELERATE))

    joint = ThreePlayerGame(
        cost_terms={Player.L: PLACEHOLDER_TERMS, Player.R: PLACEHOLDER_TERMS},
        parameters={Player.L: parameters, Player.R: parameters},
    )

    print("Full game A--L--R")
    for profile, costs in joint.cost_table(trial, av_action=Action.MERGE).items():
        print(
            f"  ({profile.left.value:8s}, {profile.rear.value:8s}) -> "
            f"C_L={costs[Player.L]:.3f}, C_R={costs[Player.R]:.3f}"
        )
    equilibria = joint.pure_nash_equilibria(trial, av_action=Action.MERGE)
    print(f"  pure equilibria: {[(p.left.value, p.rear.value) for p in equilibria]}")
    demonstrate_remix()


def demonstrate_remix() -> None:
    """Brief formulas on invented kinematics. Confirms a remix changes the cost."""
    counts = available_terms()
    if any(len(names) != 5 for names in counts.values()):
        raise AssertionError(f"Expected 5 candidates per component, got {counts}.")

    empty = TrialData()
    try:
        compose().safety(Player.L, JointAction(left=Action.YIELD), empty)
    except MissingCostFeature:
        pass
    else:
        raise AssertionError("A cost term ran without projected kinematics.")

    projected = {
        "L": {
            "Yield": {"ttc": 4.0, "speed_after": 18.0, "acceleration": -1.5, "offered_gap": 12.0},
            "Continue": {"ttc": 1.2, "speed_after": 25.0, "acceleration": 0.0, "offered_gap": 3.0},
        }
    }
    trial = TrialData(
        participant_id="P001",
        left_human=VehicleKinematics(desired_speed=25.0),
        extra={"projected": projected, "sufficient_gap_m": 8.0},
    )
    parameters = PlayerParameters(
        weights=CostWeights(safety=1.0, efficiency=1.0, comfort=1.0, interaction=1.0),
        alpha=1.0,
    )
    brief = compose(interaction="insufficient_gap")
    costs = game_al(brief, parameters).costs(trial, av_action=Action.SIGNAL)
    # 1/4 + (25-18)^2 + 1.5 + 0, and 1/1.2 + 0 + 0 + 1.
    if abs(costs[Action.YIELD] - 50.75) > 1e-9:
        raise AssertionError(costs[Action.YIELD])
    if abs(costs[Action.CONTINUE] - (1.0 / 1.2 + 1.0)) > 1e-9:
        raise AssertionError(costs[Action.CONTINUE])

    linear = compose(efficiency="speed_loss", interaction="insufficient_gap")
    swapped = game_al(linear, parameters).costs(trial, av_action=Action.SIGNAL)
    # Same trial, but the speed term is 7 m/s rather than 49 (m/s)^2.
    if abs(swapped[Action.YIELD] - 8.75) > 1e-9:
        raise AssertionError(swapped[Action.YIELD])

    print("Cost-term remix, invented kinematics")
    print(
        "  brief formulas:  "
        f"Yield {costs[Action.YIELD]:.3f}, Continue {costs[Action.CONTINUE]:.3f}"
    )
    print(f"  linear speed loss: Yield {swapped[Action.YIELD]:.3f}")


if __name__ == "__main__":
    main()
