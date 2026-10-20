"""One time series per merge, aligned to the AV cue.

The game's TrialData is a single decision instant. The HMM needs the
frames around that instant, so it keeps its own container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# Continuous channels named in the brief. Drop any that the logs do not
# actually contain. TTC is omitted here because it is built from gap and
# relative speed; adding it as well double-counts those two.
DEFAULT_FEATURES: tuple[str, ...] = (
    "speed_l",
    "accel_l",
    "gap_ahead_of_l",
    "relative_speed_l_a",
    "speed_r",
    "accel_r",
    "gap_r_to_a",
    "av_indicator",
    "av_lateral",
)

# The brief also lists the human action and stress. Both are optional.
# The action is decided once: do not copy the final choice onto earlier
# frames. Stress belongs here only when it shares the kinematic clock;
# otherwise it stays in the profile model.
OPTIONAL_FEATURES: tuple[str, ...] = (
    "ttc_l_a",
    "speed_a",
    "accel_a",
    "action_l",
    "action_r",
    "stress",
)


class MissingObservation(ValueError):
    """A frame is missing a feature the model was asked to use."""


@dataclass
class Frame:
    """One resampled instant. ``time`` is seconds relative to the AV cue."""

    time: float
    values: Mapping[str, float]


@dataclass
class TrialSequence:
    """The observation sequence for one merge."""

    frames: list[Frame]
    participant_id: str | None = None
    trial_number: int | None = None
    exposure_number: int | None = None
    frame_dt: float | None = None

    def times(self) -> np.ndarray:
        return np.asarray([frame.time for frame in self.frames], dtype=float)

    def matrix(self, feature_names: Sequence[str]) -> np.ndarray:
        """Rows are frames, columns follow ``feature_names``."""
        if not self.frames:
            raise ValueError("Trial sequence has no frames.")
        rows: list[list[float]] = []
        for frame in self.frames:
            try:
                rows.append([float(frame.values[name]) for name in feature_names])
            except KeyError as exc:
                raise MissingObservation(
                    f"Frame at t={frame.time} is missing feature {exc}."
                ) from exc
        values = np.asarray(rows, dtype=float)
        if not np.isfinite(values).all():
            raise MissingObservation("Observation matrix contains a non-finite value.")
        return values

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, float]],
        features: Sequence[str],
        *,
        time_key: str = "time",
        participant_id: str | None = None,
        trial_number: int | None = None,
        exposure_number: int | None = None,
        frame_dt: float | None = None,
    ) -> TrialSequence:
        """Build a sequence from one mapping per frame."""
        try:
            frames = [
                Frame(
                    time=float(record[time_key]),
                    values={name: record[name] for name in features},
                )
                for record in records
            ]
        except KeyError as exc:
            raise MissingObservation(f"A frame record is missing {exc}.") from exc
        return cls(
            frames=frames,
            participant_id=participant_id,
            trial_number=trial_number,
            exposure_number=exposure_number,
            frame_dt=frame_dt,
        )


@dataclass(frozen=True)
class DecodedTrial:
    """Viterbi phase at each frame. ``times`` lines up with ``phases``."""

    times: np.ndarray
    phases: tuple[str, ...]
    log_probability: float
    participant_id: str | None = None
    trial_number: int | None = None
    exposure_number: int | None = None


def decode_trial(model, trial: TrialSequence) -> DecodedTrial:
    """Most likely phase path for one trial, in the model's feature order."""
    phases, score = model.viterbi(trial.matrix(model.feature_names))
    return DecodedTrial(
        times=trial.times(),
        phases=tuple(phases),
        log_probability=score,
        participant_id=trial.participant_id,
        trial_number=trial.trial_number,
        exposure_number=trial.exposure_number,
    )


def decode_trials(model, trials: Sequence[TrialSequence]) -> list[DecodedTrial]:
    return [decode_trial(model, trial) for trial in trials]


def fit_trials(model, trials: Sequence[TrialSequence], **kwargs) -> list[float]:
    """Fit on these trials. The stored scale is theirs; reuse the same object to score others."""
    return model.fit([trial.matrix(model.feature_names) for trial in trials], **kwargs)


def first_entry(decoded: DecodedTrial, phase: str) -> float | None:
    """Cue-relative time of the first frame labelled ``phase``."""
    for time, name in zip(decoded.times, decoded.phases):
        if name == phase:
            return float(time)
    return None


def occupancy_seconds(decoded: DecodedTrial) -> dict[str, float]:
    """Seconds spent in each phase, using the gaps between frame times.

    The last frame is given the same width as the step before it. Compare
    trials only when they share a frame step.
    """
    times = np.asarray(decoded.times, dtype=float)
    if len(times) == 0:
        return {}
    if len(times) == 1:
        return {decoded.phases[0]: 0.0}
    steps = np.diff(times)
    widths = np.concatenate([steps, steps[-1:]])
    totals: dict[str, float] = {}
    for phase, width in zip(decoded.phases, widths):
        totals[phase] = totals.get(phase, 0.0) + float(width)
    return totals
