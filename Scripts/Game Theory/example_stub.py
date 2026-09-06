"""Usage sketch with placeholder cost components.

Run from anywhere:
    python "Scripts/Game Theory/example_stub.py"

The cost components below return constants. Replace them with the example
formulas from the brief, or with whatever the dataset supports, once the
trial data are available.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from costs import CostTermFunctions
from data import TrialData
from game_types import Action, CostWeights, Player, PlayerParameters
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


if __name__ == "__main__":
    main()
