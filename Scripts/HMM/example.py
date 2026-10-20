"""Checks for the HMM building blocks. Uses synthetic sequences, not trial data.

Run from anywhere:
    python "Scripts/HMM/example.py"
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exposure import split_transitions
from gaussian_hmm import GaussianHMM, merge_hmm
from phases import PHASES
from sequences import DEFAULT_FEATURES, DecodedTrial, first_entry, occupancy_seconds


def _known_two_state() -> GaussianHMM:
    model = GaussianHMM(
        state_names=("A", "B"),
        feature_names=("x",),
        transition_mask=np.array([[False, True], [True, True]]),
        start_mask=np.array([True, True]),
    )
    model.means = np.array([[0.0], [3.0]])
    model.variances = np.array([[1.0], [1.0]])
    model.log_start = np.log(np.array([0.6, 0.4]))
    model.log_transition = np.full((2, 2), -np.inf)
    model.log_transition[0, 1] = np.log(1.0)
    model.log_transition[1, 0] = np.log(0.3)
    model.log_transition[1, 1] = np.log(0.7)
    return model


def check_forward() -> None:
    """Forward and Viterbi against every phase path of a 2-state, 3-frame model."""
    model = _known_two_state()
    observations = np.array([[0.2], [2.5], [3.1]])
    scaled = model._scale(observations)
    log_emission = model._log_emissions(scaled)
    log_alpha, _ = model._forward(log_emission)
    log_beta = model._backward(log_emission)
    _, log_likelihood = model._forward(log_emission)
    gamma = np.exp(log_alpha + log_beta - log_likelihood)
    if not np.allclose(gamma.sum(axis=1), 1.0):
        raise AssertionError("Phase posteriors do not sum to one.")

    totals = []
    best_path: tuple[int, ...] | None = None
    best_score = -np.inf
    for path in product(range(model.n_states), repeat=len(observations)):
        score = model.log_start[path[0]] + log_emission[0, path[0]]
        for time in range(1, len(path)):
            score = (
                score
                + model.log_transition[path[time - 1], path[time]]
                + log_emission[time, path[time]]
            )
        totals.append(score)
        if score > best_score:
            best_score = float(score)
            best_path = path
    peak = max(totals)
    brute = float(peak + np.log(np.sum(np.exp(np.asarray(totals) - peak))))
    got = model.log_likelihood(observations)
    if abs(got - brute) > 1e-6:
        raise AssertionError(f"Forward {got} disagrees with enumeration {brute}.")
    decoded, _ = model.viterbi(observations)
    expected = [model.state_names[index] for index in best_path]
    if decoded != expected:
        raise AssertionError(f"Viterbi {decoded} disagrees with {expected}.")
    print("Forward check: log-likelihood and Viterbi match enumeration")


def _chain() -> GaussianHMM:
    model = GaussianHMM(
        state_names=("S0", "S1", "S2"),
        feature_names=("x", "y"),
        transition_mask=np.array(
            [[True, True, False], [False, True, True], [False, False, True]]
        ),
        start_mask=np.array([True, False, False]),
    )
    model.means = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]])
    model.variances = np.full((3, 2), 0.25)
    model.log_start = np.array([0.0, -np.inf, -np.inf])
    stay, leave = np.log(0.75), np.log(0.25)
    model.log_transition = np.array(
        [
            [stay, leave, -np.inf],
            [-np.inf, stay, leave],
            [-np.inf, -np.inf, 0.0],
        ]
    )
    return model


def check_fit() -> None:
    """Baum-Welch on a known 3-phase chain should raise the likelihood and recover it."""
    source = _chain()
    rng = np.random.default_rng(0)
    generated = [source.sample(30, rng) for _ in range(40)]
    fitted = GaussianHMM(
        state_names=("S0", "S1", "S2"),
        feature_names=("x", "y"),
        transition_mask=source.transition_mask.copy(),
        start_mask=source.start_mask.copy(),
    )
    history = fitted.fit([obs for obs, _path in generated], n_iter=40, seed=0)
    for earlier, later in zip(history, history[1:]):
        if later < earlier - 1e-3 * (1.0 + abs(earlier)):
            raise AssertionError(f"Log-likelihood fell from {earlier} to {later}.")
    if fitted.transition_matrix()[0, 2] != 0.0:
        raise AssertionError("A forbidden transition received probability.")

    correct = 0
    total = 0
    for observations, truth in generated:
        decoded, _score = fitted.viterbi(observations)
        correct += sum(left == right for left, right in zip(decoded, truth))
        total += len(truth)
    accuracy = correct / total
    recovered = fitted.emission_means()
    error = max(
        abs(recovered[phase][feature] - source.means[index, feature_index])
        for index, phase in enumerate(("S0", "S1", "S2"))
        for feature_index, feature in enumerate(("x", "y"))
    )
    print(
        f"Fit check: log-likelihood {history[0]:.1f} -> {history[-1]:.1f}, "
        f"accuracy {accuracy:.2f}, mean error {error:.3f}"
    )
    if accuracy < 0.8:
        raise AssertionError(f"Viterbi accuracy {accuracy:.2f} is below 0.8.")
    if error > 0.5:
        raise AssertionError(f"Emission-mean error {error:.3f} is above 0.5.")


def check_merge_mask() -> None:
    model = merge_hmm(DEFAULT_FEATURES)
    if model.state_names != PHASES:
        raise AssertionError("Merge phases drifted from the brief.")
    index = {name: i for i, name in enumerate(PHASES)}
    if model.transition_mask[index["Resolution"], index["Approach"]]:
        raise AssertionError("Resolution should not return to Approach.")
    for phase in ("Yield", "Continue", "Conflict"):
        if not model.transition_mask[index["Anticipation"], index[phase]]:
            raise AssertionError(f"Anticipation should be able to enter {phase}.")
    if model.feature_names != DEFAULT_FEATURES:
        raise AssertionError("Default features were not stored on the model.")
    rng = np.random.default_rng(1)
    smoke = merge_hmm(("x", "y"))
    history = smoke.fit([rng.normal(size=(20, 2)) for _ in range(4)], n_iter=3, seed=1)
    if not np.isfinite(history[-1]):
        raise AssertionError("Six-phase fit produced a non-finite likelihood.")
    if smoke.transition_matrix()[index["Resolution"], index["Approach"]] != 0.0:
        raise AssertionError("Fit wrote a forbidden transition.")
    print("Merge mask: 6 phases, Anticipation branches, Resolution stays")


def check_exposure() -> None:
    split = split_transitions(
        exposures=[1, None, 4],
        paths=[
            ["Approach", "Anticipation", "Yield"],
            ["Approach", "Approach"],
            ["Yield", "Resolution"],
        ],
        state_names=PHASES,
        early_max=1,
    )
    if (split.n_early, split.n_later, split.skipped) != (1, 1, 1):
        raise AssertionError("Exposure split counted the paths wrong.")
    approach = PHASES.index("Approach")
    anticipation = PHASES.index("Anticipation")
    if split.early_counts[approach, anticipation] != 1:
        raise AssertionError("Early transition count is wrong.")
    decoded = DecodedTrial(
        times=np.array([0.0, 0.1, 0.2]),
        phases=("Approach", "Approach", "Anticipation"),
        log_probability=0.0,
    )
    if first_entry(decoded, "Anticipation") != 0.2:
        raise AssertionError("first_entry missed the phase onset.")
    occupied = occupancy_seconds(decoded)
    if abs(occupied["Approach"] - 0.2) > 1e-9 or abs(occupied["Anticipation"] - 0.1) > 1e-9:
        raise AssertionError(f"occupancy_seconds returned {occupied}.")
    print(
        f"Exposure split: {split.n_early} early, {split.n_later} later, "
        f"{split.skipped} skipped"
    )


def main() -> None:
    check_forward()
    check_fit()
    check_merge_mask()
    check_exposure()


if __name__ == "__main__":
    main()
