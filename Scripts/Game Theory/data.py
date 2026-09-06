"""Trial data container - fill in when experimental data are available."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VehicleKinematics:
    """Placeholder kinematics for one vehicle at a decision-relevant instant.

    Replace or extend the fields once the processed dataset schema is fixed.
    """

    position: float | None = None
    speed: float | None = None
    acceleration: float | None = None
    desired_speed: float | None = None
    bumper_gap_ahead: float | None = None
    bumper_gap_behind: float | None = None
    relative_speed: float | None = None
    ttc: float | None = None
    time_headway: float | None = None
    required_braking: float | None = None


@dataclass
class TrialData:
    """Empty container for one merge trial or decision window.

    The cost components are all example formulas in the brief, so this holds
    the raw and derived quantities they might use rather than committing to
    any one definition.
    """

    participant_id: str | None = None
    trial_number: int | None = None
    scenario_id: str | None = None

    av: VehicleKinematics = field(default_factory=VehicleKinematics)
    left_human: VehicleKinematics = field(default_factory=VehicleKinematics)
    rear_human: VehicleKinematics = field(default_factory=VehicleKinematics)

    av_indicator_on: bool | None = None
    av_lateral_movement: float | None = None
    av_cue_time: float | None = None
    av_strategy: str | None = None

    left_response_time: float | None = None
    rear_response_time: float | None = None

    observed_action_av: str | None = None
    observed_action_left: str | None = None
    observed_action_rear: str | None = None

    # Human factors, usable only if measured before the predicted decision.
    stress_pre_decision: float | None = None
    exposure_number: int | None = None
    previous_outcome: str | None = None
    previous_action_left: str | None = None
    previous_action_rear: str | None = None
    previous_reaction_time: float | None = None
    gaze_features: dict[str, Any] = field(default_factory=dict)

    extra: dict[str, Any] = field(default_factory=dict)
    """Derived features not yet modelled explicitly, such as the projected
    speed, acceleration, or gap under each candidate action."""

    def reaction_time(self, side: str = "left") -> float | None:
        """Response time minus AV cue time, for ``left`` or ``rear``."""
        response = self.left_response_time if side == "left" else self.rear_response_time
        if self.av_cue_time is None or response is None:
            return None
        return response - self.av_cue_time
