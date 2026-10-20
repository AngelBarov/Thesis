"""Compare phase-to-phase moves across repeated AV exposures.

Exposure does not enter the transition matrix. The profile model is where
exposure and stress predict a response. Here the decoded paths are only
counted, early exposures against later ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from .sequences import DecodedTrial
except ImportError:
    from sequences import DecodedTrial


@dataclass(frozen=True)
class ExposureSplit:
    """Transition counts for exposures <= early_max, and the rest."""

    early_max: int
    early_counts: np.ndarray
    later_counts: np.ndarray
    early_probabilities: np.ndarray
    later_probabilities: np.ndarray
    n_early: int
    n_later: int
    skipped: int


def transition_counts(
    paths: Sequence[Sequence[str]], state_names: Sequence[str]
) -> np.ndarray:
    """Count observed moves. Rows and columns follow ``state_names``."""
    index = {name: i for i, name in enumerate(state_names)}
    counts = np.zeros((len(index), len(index)), dtype=float)
    for path in paths:
        for source, target in zip(path, path[1:]):
            try:
                counts[index[source], index[target]] += 1.0
            except KeyError as exc:
                raise ValueError(f"Phase {exc} is not in the phase set.") from exc
    return counts


def probabilities_from_counts(counts: np.ndarray) -> np.ndarray:
    """Row-normalise counts. A row of zeros means that phase was never left."""
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.zeros_like(counts, dtype=float)
    np.divide(counts, totals, out=probabilities, where=totals > 0.0)
    return probabilities


def split_transitions(
    exposures: Sequence[int | None],
    paths: Sequence[Sequence[str]],
    state_names: Sequence[str],
    early_max: int,
) -> ExposureSplit:
    """Split decoded paths into early and later exposures.

    An exposure of ``None`` is skipped. ``early_max`` is inclusive.
    """
    if len(exposures) != len(paths):
        raise ValueError("exposures and paths must have the same length.")
    early: list[Sequence[str]] = []
    later: list[Sequence[str]] = []
    skipped = 0
    for exposure, path in zip(exposures, paths):
        if exposure is None:
            skipped += 1
            continue
        if exposure <= early_max:
            early.append(path)
        else:
            later.append(path)
    early_counts = transition_counts(early, state_names)
    later_counts = transition_counts(later, state_names)
    return ExposureSplit(
        early_max=early_max,
        early_counts=early_counts,
        later_counts=later_counts,
        early_probabilities=probabilities_from_counts(early_counts),
        later_probabilities=probabilities_from_counts(later_counts),
        n_early=len(early),
        n_later=len(later),
        skipped=skipped,
    )


def split_decoded(
    decoded: Sequence[DecodedTrial],
    state_names: Sequence[str],
    early_max: int,
) -> ExposureSplit:
    """split_transitions on the paths already decoded for each trial."""
    return split_transitions(
        [trial.exposure_number for trial in decoded],
        [trial.phases for trial in decoded],
        state_names,
        early_max,
    )
