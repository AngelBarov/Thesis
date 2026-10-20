"""Candidate formulas for the four cost components.

The weighted sum is in costs.py. This module only supplies the formulas.
compose() picks one per component, so L and R can use different remixes.

The formulas do not simulate a trajectory. Every action-conditional number
is read from TrialData.extra["projected"][player][action][key], where
player and action are the enum values ("L", "Yield"). desired_speed stays
on the vehicle. The projection step, once the data exist, decides which
physical gap or braking demand each key refers to.
"""

from __future__ import annotations

import math
from typing import Mapping

try:
    from .costs import CostTermFunctions
    from .data import TrialData, VehicleKinematics
    from .game_types import Action, JointAction, Player
except ImportError:
    from costs import CostTermFunctions
    from data import TrialData, VehicleKinematics
    from game_types import Action, JointAction, Player

# Floors for a zero gap, time, or speed. Not estimated.
EPS_TIME = 0.1
EPS_GAP = 0.1
EPS_SPEED = 0.1

# Shape constants. A weight cannot absorb these, because they sit inside
# an exponential or a threshold. Override from TrialData.extra with the
# same name. Fix them before fitting; do not tune them to raise the hit rate.
TTC_SCALE_S = 1.5
COMFORT_ACCEL_MPS2 = 1.5
REFERENCE_DISTANCE_M = 20.0

PROJECTED_KEYS: tuple[str, ...] = (
    "ttc",
    "gap",
    "time_headway",
    "required_braking",
    "speed_after",
    "acceleration",
    "jerk",
    "time_below_desired",
    "distance_lost",
    "offered_gap",
    "gap_change",
    "imposed_braking",
)


class MissingCostFeature(ValueError):
    """An input a cost term needs is not on the trial."""


def _finite(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissingCostFeature(f"{description} is not a number.")
    number = float(value)
    if not math.isfinite(number):
        raise MissingCostFeature(f"{description} is not finite.")
    return number


def _action(player: Player, joint_action: JointAction) -> Action:
    action = joint_action.of(player)
    if action is None:
        raise ValueError(f"{player.value} has no action in this combination.")
    return action


def _projected(data: TrialData, player: Player, action: Action, key: str) -> float:
    projected = data.extra.get("projected")
    location = f"projected[{player.value!r}][{action.value!r}][{key!r}]"
    if not isinstance(projected, Mapping):
        raise MissingCostFeature(
            "TrialData.extra['projected'] is missing. Map player to action to "
            "feature before evaluating a cost term."
        )
    try:
        value = projected[player.value][action.value][key]
    except (KeyError, TypeError) as exc:
        raise MissingCostFeature(f"Missing {location}.") from exc
    return _finite(value, location)


def _vehicle(data: TrialData, player: Player) -> VehicleKinematics:
    return {
        Player.A: data.av,
        Player.L: data.left_human,
        Player.R: data.rear_human,
    }[player]


def _desired_speed(data: TrialData, player: Player) -> float:
    value = _vehicle(data, player).desired_speed
    if value is None:
        raise MissingCostFeature(
            f"{player.value} has no desired_speed. Set it on the vehicle kinematics."
        )
    return _finite(value, "desired_speed")


def _speeds(player: Player, joint_action: JointAction, data: TrialData) -> tuple[float, float]:
    action = _action(player, joint_action)
    return _desired_speed(data, player), _projected(data, player, action, "speed_after")


def _extra(data: TrialData, key: str, default: float | None) -> float:
    if key not in data.extra or data.extra[key] is None:
        if default is None:
            raise MissingCostFeature(
                f"Set TrialData.extra[{key!r}] before using this cost term."
            )
        return default
    number = _finite(data.extra[key], key)
    if number <= 0.0:
        raise MissingCostFeature(f"{key} must be positive.")
    return number


def _non_negative_time(value: float, name: str) -> float:
    if value < 0.0:
        raise ValueError(
            f"{name} is negative ({value}). When the gap is opening, pass a large {name}."
        )
    return value


def inverse_ttc(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """1 / max(TTC, epsilon). Brief example safety cost."""
    ttc = _non_negative_time(
        _projected(data, player, _action(player, joint_action), "ttc"), "ttc"
    )
    return 1.0 / max(ttc, EPS_TIME)


def inverse_gap(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """1 / max(gap, epsilon). Projected minimum gap, as an inverse-TTC analogue."""
    gap = _projected(data, player, _action(player, joint_action), "gap")
    return 1.0 / max(gap, EPS_GAP)


def required_braking(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """max(0, required braking). Zero when the action needs no braking."""
    demand = _projected(data, player, _action(player, joint_action), "required_braking")
    return max(0.0, demand)


def inverse_time_headway(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """1 / max(time headway, epsilon). Ignores relative speed, unlike TTC."""
    headway = _non_negative_time(
        _projected(data, player, _action(player, joint_action), "time_headway"),
        "time_headway",
    )
    return 1.0 / max(headway, EPS_TIME)


def exponential_ttc(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """exp(-TTC / tau). Same input as inverse TTC, but bounded in (0, 1]."""
    ttc = _non_negative_time(
        _projected(data, player, _action(player, joint_action), "ttc"), "ttc"
    )
    tau = _extra(data, "ttc_scale_s", TTC_SCALE_S)
    return math.exp(-ttc / tau)


def squared_speed_loss(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """(v_desired - v_after)^2. Brief example. A speed gain is charged as well."""
    desired, after = _speeds(player, joint_action, data)
    return (desired - after) ** 2


def speed_loss(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """max(0, v_desired - v_after). Slowing down only, without the square."""
    desired, after = _speeds(player, joint_action, data)
    return max(0.0, desired - after)


def time_delay(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """Extra seconds to cover a reference distance at v_after instead of v_desired.

    The distance scales the term and the efficiency weight absorbs that scale.
    """
    desired, after = _speeds(player, joint_action, data)
    distance = _extra(data, "reference_distance_m", REFERENCE_DISTANCE_M)
    delay = distance / max(after, EPS_SPEED) - distance / max(desired, EPS_SPEED)
    return max(0.0, delay)


def time_below_desired(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """Seconds spent below desired speed. The projection supplies the integral."""
    duration = _projected(data, player, _action(player, joint_action), "time_below_desired")
    return max(0.0, duration)


def distance_lost(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """Metres of progress given up. The projection supplies the distance."""
    lost = _projected(data, player, _action(player, joint_action), "distance_lost")
    return max(0.0, lost)


def abs_acceleration(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """|a|. Brief example comfort cost."""
    accel = _projected(data, player, _action(player, joint_action), "acceleration")
    return abs(accel)


def squared_acceleration(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """a^2. Same input as |a|, with a heavier charge on strong acceleration."""
    accel = _projected(data, player, _action(player, joint_action), "acceleration")
    return accel * accel


def abs_jerk(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """|jerk|. Use only when the derivative is trustworthy."""
    jerk = _projected(data, player, _action(player, joint_action), "jerk")
    return abs(jerk)


def squared_jerk(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """jerk^2. Same caveat as abs_jerk."""
    jerk = _projected(data, player, _action(player, joint_action), "jerk")
    return jerk * jerk


def harsh_acceleration(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """max(0, |a| - a_comfort). No cost inside the comfort band."""
    accel = abs(_projected(data, player, _action(player, joint_action), "acceleration"))
    limit = _extra(data, "comfort_accel_mps2", COMFORT_ACCEL_MPS2)
    return max(0.0, accel - limit)


def insufficient_gap(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """1 when the gap left for the AV is below sufficient_gap_m, else 0."""
    offered = _projected(data, player, _action(player, joint_action), "offered_gap")
    return 0.0 if offered >= _extra(data, "sufficient_gap_m", None) else 1.0


def gap_shortfall(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """max(0, sufficient_gap_m - offered gap). Continuous version of the binary gap."""
    offered = _projected(data, player, _action(player, joint_action), "offered_gap")
    return max(0.0, _extra(data, "sufficient_gap_m", None) - offered)


def gap_closure(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """max(0, -gap_change). How much the action closes the gap offered to the AV."""
    change = _projected(data, player, _action(player, joint_action), "gap_change")
    return max(0.0, -change)


def imposed_braking(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """max(0, braking the action forces on another vehicle)."""
    imposed = _projected(data, player, _action(player, joint_action), "imposed_braking")
    return max(0.0, imposed)


def inverse_offered_gap(player: Player, joint_action: JointAction, data: TrialData) -> float:
    """1 / max(offered gap, epsilon). No sufficiency threshold."""
    offered = _projected(data, player, _action(player, joint_action), "offered_gap")
    return 1.0 / max(offered, EPS_GAP)


SAFETY_TERMS = {
    "inverse_ttc": inverse_ttc,
    "inverse_gap": inverse_gap,
    "required_braking": required_braking,
    "inverse_time_headway": inverse_time_headway,
    "exponential_ttc": exponential_ttc,
}

EFFICIENCY_TERMS = {
    "squared_speed_loss": squared_speed_loss,
    "speed_loss": speed_loss,
    "time_delay": time_delay,
    "time_below_desired": time_below_desired,
    "distance_lost": distance_lost,
}

COMFORT_TERMS = {
    "abs_acceleration": abs_acceleration,
    "squared_acceleration": squared_acceleration,
    "abs_jerk": abs_jerk,
    "squared_jerk": squared_jerk,
    "harsh_acceleration": harsh_acceleration,
}

INTERACTION_TERMS = {
    "insufficient_gap": insufficient_gap,
    "gap_shortfall": gap_shortfall,
    "gap_closure": gap_closure,
    "imposed_braking": imposed_braking,
    "inverse_offered_gap": inverse_offered_gap,
}


def available_terms() -> dict[str, tuple[str, ...]]:
    """Names that compose() accepts, in component order."""
    return {
        "safety": tuple(SAFETY_TERMS),
        "efficiency": tuple(EFFICIENCY_TERMS),
        "comfort": tuple(COMFORT_TERMS),
        "interaction": tuple(INTERACTION_TERMS),
    }


def _pick(catalog: Mapping[str, object], name: str, component: str):
    try:
        return catalog[name]
    except KeyError:
        known = ", ".join(catalog)
        raise ValueError(f"Unknown {component} term {name!r}. Choose from: {known}.") from None


def compose(
    safety: str = "inverse_ttc",
    efficiency: str = "squared_speed_loss",
    comfort: str = "abs_acceleration",
    interaction: str | None = None,
) -> CostTermFunctions:
    """Build the cost terms for one player.

    Defaults are the three example formulas in the brief. Pass
    interaction="insufficient_gap" for the brief's optional gap indicator.
    That term needs extra["sufficient_gap_m"].
    """
    return CostTermFunctions(
        safety=_pick(SAFETY_TERMS, safety, "safety"),
        efficiency=_pick(EFFICIENCY_TERMS, efficiency, "efficiency"),
        comfort=_pick(COMFORT_TERMS, comfort, "comfort"),
        interaction=None
        if interaction is None
        else _pick(INTERACTION_TERMS, interaction, "interaction"),
    )
