"""Diagonal-Gaussian HMM for one merge.

Rabiner's three calculations, with a Gaussian emission per phase instead
of a discrete symbol: binning speed and gap would throw the scale away.
``fit`` is Baum-Welch, ``log_likelihood`` is the forward pass, and
``viterbi`` is the most likely phase path.

Features are standardised inside ``fit`` from the sequences passed in.
Score later trials with the same object so they use that training scale.
Forbidden transitions stay at probability zero. A small pseudocount on
the allowed edges keeps an unused edge from vanishing on the first update.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from .phases import DEFAULT_START, DEFAULT_TRANSITIONS, PHASES
except ImportError:
    from phases import DEFAULT_START, DEFAULT_TRANSITIONS, PHASES

# Variance floor is in standardised units, where each feature has spread 1.
VAR_FLOOR = 1e-3
TRANSITION_PSEUDOCOUNT = 1e-3


def _log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    logged = np.full(probabilities.shape, -np.inf, dtype=float)
    positive = probabilities > 0.0
    logged[positive] = np.log(probabilities[positive])
    return logged


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis)
    finite = np.isfinite(max_value)
    safe_max = np.where(finite, max_value, 0.0)
    summed = np.sum(np.exp(values - np.expand_dims(safe_max, axis=axis)), axis=axis)
    with np.errstate(divide="ignore"):
        logged = np.log(summed)
    result = np.where(summed > 0.0, safe_max + logged, -np.inf)
    return np.where(finite, result, -np.inf)


class GaussianHMM:
    """Phases with diagonal-Gaussian emissions and an optional transition mask."""

    def __init__(
        self,
        state_names: Sequence[str],
        feature_names: Sequence[str],
        transition_mask: np.ndarray | None = None,
        start_mask: np.ndarray | None = None,
    ) -> None:
        self.state_names = tuple(state_names)
        self.feature_names = tuple(feature_names)
        self.n_states = len(self.state_names)
        self.n_features = len(self.feature_names)
        if self.n_states == 0 or self.n_features == 0:
            raise ValueError("The model needs at least one phase and one feature.")
        if len(set(self.state_names)) != self.n_states:
            raise ValueError("Phase names must be unique.")
        if len(set(self.feature_names)) != self.n_features:
            raise ValueError("Feature names must be unique.")

        if transition_mask is None:
            transition_mask = np.ones((self.n_states, self.n_states), dtype=bool)
        if start_mask is None:
            start_mask = np.ones(self.n_states, dtype=bool)
        self.transition_mask = np.array(transition_mask, dtype=bool, copy=True)
        self.start_mask = np.array(start_mask, dtype=bool, copy=True)
        if self.transition_mask.shape != (self.n_states, self.n_states):
            raise ValueError("transition_mask must have shape (n_phases, n_phases).")
        if self.start_mask.shape != (self.n_states,):
            raise ValueError("start_mask must have one entry per phase.")
        if not self.transition_mask.any(axis=1).all():
            raise ValueError("Every phase needs at least one allowed transition.")
        if not self.start_mask.any():
            raise ValueError("At least one phase must be allowed as a start.")

        self.means: np.ndarray | None = None
        self.variances: np.ndarray | None = None
        self.log_start: np.ndarray | None = None
        self.log_transition: np.ndarray | None = None
        self.feature_mean = np.zeros(self.n_features)
        self.feature_std = np.ones(self.n_features)
        self.history: list[float] = []

    def fit(
        self,
        observations: Sequence[np.ndarray],
        *,
        n_iter: int = 40,
        tol: float = 1e-4,
        seed: int = 0,
        standardise: bool = True,
    ) -> list[float]:
        """Estimate start, transitions, and emissions. Returns the log-likelihood trace.

        The first value is the initialisation. Later values are measured
        after each Baum-Welch update. Stops once the gain falls below ``tol``.
        """
        sequences = [self._as_matrix(values) for values in observations]
        if not sequences:
            raise ValueError("Fit needs at least one sequence.")
        scaled = self._fit_scale(sequences, standardise)
        rng = np.random.default_rng(seed)
        self._initialise(scaled, rng)

        history = [self._total_log_likelihood(scaled)]
        for _ in range(n_iter):
            self._m_step(self._statistics(scaled))
            updated = self._total_log_likelihood(scaled)
            gain = updated - history[-1]
            history.append(updated)
            if gain < tol * (1.0 + abs(history[-2])):
                break
        self.history = history
        return history

    def log_likelihood(self, observations: np.ndarray) -> float:
        """Forward log-likelihood of one sequence, in the original feature units."""
        self._require()
        scaled = self._scale(self._as_matrix(observations))
        _, total = self._forward(self._log_emissions(scaled))
        return float(total)

    def log_likelihoods(self, observations: Sequence[np.ndarray]) -> list[float]:
        return [self.log_likelihood(sequence) for sequence in observations]

    def viterbi(self, observations: np.ndarray) -> tuple[list[str], float]:
        """Most likely phase path, and its joint log-probability."""
        self._require()
        scaled = self._scale(self._as_matrix(observations))
        log_emission = self._log_emissions(scaled)
        length, n_states = log_emission.shape
        delta = np.empty((length, n_states))
        pointer = np.zeros((length, n_states), dtype=int)
        delta[0] = self.log_start + log_emission[0]
        for time in range(1, length):
            scores = delta[time - 1][:, None] + self.log_transition
            pointer[time] = np.argmax(scores, axis=0)
            delta[time] = log_emission[time] + scores[pointer[time], np.arange(n_states)]
        last = int(np.argmax(delta[-1]))
        path = [last]
        for time in range(length - 1, 0, -1):
            last = int(pointer[time, last])
            path.append(last)
        path.reverse()
        return [self.state_names[index] for index in path], float(delta[-1, path[-1]])

    def sample(
        self, length: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        """Draw one sequence in the original feature units, and its phase path."""
        self._require()
        if length < 1:
            raise ValueError("length must be positive.")
        state = self._draw(self.log_start, rng)
        states = [state]
        for _ in range(length - 1):
            state = self._draw(self.log_transition[state], rng)
            states.append(state)
        scaled = np.empty((length, self.n_features))
        for time, index in enumerate(states):
            width = np.sqrt(self.variances[index])
            scaled[time] = rng.normal(self.means[index], width)
        observations = scaled * self.feature_std + self.feature_mean
        return observations, [self.state_names[index] for index in states]

    def emission_means(self) -> dict[str, dict[str, float]]:
        """Emission mean of each feature in each phase, original units."""
        self._require()
        raw = self.means * self.feature_std + self.feature_mean
        return {
            phase: {
                feature: float(raw[phase_index, feature_index])
                for feature_index, feature in enumerate(self.feature_names)
            }
            for phase_index, phase in enumerate(self.state_names)
        }

    def emission_variances(self) -> dict[str, dict[str, float]]:
        """Emission variance of each feature in each phase, original units."""
        self._require()
        raw = self.variances * (self.feature_std ** 2)
        return {
            phase: {
                feature: float(raw[phase_index, feature_index])
                for feature_index, feature in enumerate(self.feature_names)
            }
            for phase_index, phase in enumerate(self.state_names)
        }

    def transition_matrix(self) -> np.ndarray:
        """Rows sum to one. A forbidden edge is exactly zero."""
        self._require()
        return np.exp(self.log_transition)

    def start_probabilities(self) -> np.ndarray:
        self._require()
        return np.exp(self.log_start)

    def _require(self) -> None:
        if self.means is None or self.log_transition is None or self.log_start is None:
            raise RuntimeError("Fit the model, or set its parameters, before using it.")
        assert self.variances is not None

    def _as_matrix(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(f"Expected shape (T, {self.n_features}), got {values.shape}.")
        if len(values) == 0:
            raise ValueError("A sequence has no frames.")
        if not np.isfinite(values).all():
            raise ValueError("Observations contain a non-finite value.")
        return values

    def _scale(self, values: np.ndarray) -> np.ndarray:
        return (values - self.feature_mean) / self.feature_std

    def _fit_scale(self, sequences: list[np.ndarray], standardise: bool) -> list[np.ndarray]:
        if standardise:
            pooled = np.concatenate(sequences, axis=0)
            self.feature_mean = pooled.mean(axis=0)
            spread = pooled.std(axis=0)
            self.feature_std = np.where(spread < 1e-8, 1.0, spread)
        else:
            self.feature_mean = np.zeros(self.n_features)
            self.feature_std = np.ones(self.n_features)
        return [self._scale(sequence) for sequence in sequences]

    def _initialise(self, sequences: list[np.ndarray], rng: np.random.Generator) -> None:
        pooled = np.concatenate(sequences, axis=0)
        variance = np.maximum(pooled.var(axis=0), VAR_FLOOR)
        self.means = self._temporal_means(sequences, pooled.mean(axis=0), rng)
        self.variances = np.tile(variance, (self.n_states, 1))

        start = np.zeros(self.n_states)
        start[self.start_mask] = 1.0
        self.log_start = _log_probabilities(start / start.sum())
        transition = np.zeros((self.n_states, self.n_states))
        for source in range(self.n_states):
            allowed = self.transition_mask[source]
            transition[source, allowed] = 1.0 / np.count_nonzero(allowed)
        self.log_transition = _log_probabilities(transition)

    def _temporal_means(
        self,
        sequences: list[np.ndarray],
        global_mean: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Seed the first phase from early frames and the last from late frames.

        Phases in between start at the middle of the trial. When the phase
        set has Anticipation, that phase keeps the middle and the remaining
        decision phases are spread around it so Baum-Welch has distinct
        places to move.
        """
        bins: list[list[np.ndarray]] = [[], [], []]
        for sequence in sequences:
            edges = np.linspace(0, len(sequence), 4, dtype=int)
            for bin_index in range(3):
                chunk = sequence[edges[bin_index] : edges[bin_index + 1]]
                if len(chunk):
                    bins[bin_index].append(chunk)

        def centre(chunks: list[np.ndarray]) -> np.ndarray:
            if not chunks:
                return global_mean.copy()
            return np.concatenate(chunks, axis=0).mean(axis=0)

        early, middle, late = (centre(chunks) for chunks in bins)
        means = np.tile(global_mean, (self.n_states, 1))
        means[0] = early
        if self.n_states > 1:
            means[-1] = late
        middle_frames = np.concatenate(bins[1], axis=0) if bins[1] else np.concatenate(sequences, axis=0)
        spread = middle_frames.std(axis=0)
        spread = np.where(spread < 1e-3, 1.0, spread)
        middle_indexes = list(range(1, max(self.n_states - 1, 1)))
        if len(middle_indexes) == 1:
            means[middle_indexes[0]] = middle
        elif middle_indexes:
            names = self.state_names
            if "Anticipation" in names and names.index("Anticipation") in middle_indexes:
                means[names.index("Anticipation")] = middle
                decision = [index for index in middle_indexes if names[index] != "Anticipation"]
            else:
                decision = middle_indexes
            for slot, index in enumerate(decision):
                position = 0.0 if len(decision) == 1 else -1.0 + 2.0 * slot / (len(decision) - 1)
                means[index] = middle + 0.5 * position * spread
        return means + rng.normal(0.0, 0.01, size=means.shape)

    def _log_emissions(self, scaled: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.variances is not None
        log_det = -0.5 * np.sum(np.log(2.0 * np.pi * self.variances), axis=1)
        diff = scaled[:, None, :] - self.means[None, :, :]
        quad = np.sum(diff * diff / self.variances[None, :, :], axis=2)
        return log_det[None, :] + (-0.5 * quad)

    def _forward(self, log_emission: np.ndarray) -> tuple[np.ndarray, float]:
        assert self.log_start is not None and self.log_transition is not None
        length, _n_states = log_emission.shape
        log_alpha = np.empty_like(log_emission)
        log_alpha[0] = self.log_start + log_emission[0]
        for time in range(1, length):
            arriving = log_alpha[time - 1][:, None] + self.log_transition
            log_alpha[time] = log_emission[time] + _logsumexp(arriving, axis=0)
        return log_alpha, float(_logsumexp(log_alpha[-1], axis=0))

    def _backward(self, log_emission: np.ndarray) -> np.ndarray:
        assert self.log_transition is not None
        length, _n_states = log_emission.shape
        log_beta = np.zeros_like(log_emission)
        for time in range(length - 2, -1, -1):
            departing = (
                self.log_transition
                + log_emission[time + 1][None, :]
                + log_beta[time + 1][None, :]
            )
            log_beta[time] = _logsumexp(departing, axis=1)
        return log_beta

    def _statistics(self, sequences: list[np.ndarray]) -> dict[str, np.ndarray | float]:
        occupancy = np.zeros(self.n_states)
        start = np.zeros(self.n_states)
        feature_sum = np.zeros((self.n_states, self.n_features))
        feature_square_sum = np.zeros((self.n_states, self.n_features))
        transitions = np.zeros((self.n_states, self.n_states))
        total = 0.0
        for sequence in sequences:
            log_emission = self._log_emissions(sequence)
            log_alpha, log_likelihood = self._forward(log_emission)
            if not np.isfinite(log_likelihood):
                raise RuntimeError(
                    "A sequence has zero probability under the current mask and emissions."
                )
            log_beta = self._backward(log_emission)
            log_gamma = log_alpha + log_beta - log_likelihood
            frame_max = np.max(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma - frame_max)
            gamma /= gamma.sum(axis=1, keepdims=True)
            occupancy += gamma.sum(axis=0)
            start += gamma[0]
            feature_sum += gamma.T @ sequence
            feature_square_sum += gamma.T @ (sequence * sequence)
            total += log_likelihood
            if len(sequence) < 2:
                continue
            log_xi = (
                log_alpha[:-1, :, None]
                + self.log_transition[None, :, :]
                + log_emission[1:, None, :]
                + log_beta[1:, None, :]
                - log_likelihood
            )
            xi = np.exp(log_xi)
            xi[~np.isfinite(xi)] = 0.0
            transitions += xi.sum(axis=0)
        return {
            "log_likelihood": total,
            "occupancy": occupancy,
            "start": start,
            "feature_sum": feature_sum,
            "feature_square_sum": feature_square_sum,
            "transitions": transitions,
        }

    def _total_log_likelihood(self, sequences: list[np.ndarray]) -> float:
        total = 0.0
        for sequence in sequences:
            _, log_likelihood = self._forward(self._log_emissions(sequence))
            total += log_likelihood
        return float(total)

    def _m_step(self, stats: dict[str, np.ndarray | float]) -> None:
        assert self.means is not None and self.variances is not None
        occupancy = stats["occupancy"]
        feature_sum = stats["feature_sum"]
        feature_square_sum = stats["feature_square_sum"]
        assert isinstance(occupancy, np.ndarray)
        assert isinstance(feature_sum, np.ndarray)
        assert isinstance(feature_square_sum, np.ndarray)
        for phase in range(self.n_states):
            if occupancy[phase] <= 1e-8:
                continue
            mean = feature_sum[phase] / occupancy[phase]
            second = feature_square_sum[phase] / occupancy[phase]
            self.means[phase] = mean
            self.variances[phase] = np.maximum(second - mean * mean, VAR_FLOOR)

        transitions = stats["transitions"]
        start = stats["start"]
        assert isinstance(transitions, np.ndarray)
        assert isinstance(start, np.ndarray)
        updated = np.zeros((self.n_states, self.n_states))
        for source in range(self.n_states):
            allowed = self.transition_mask[source]
            updated[source, allowed] = transitions[source, allowed] + TRANSITION_PSEUDOCOUNT
            updated[source] /= updated[source].sum()
        self.log_transition = _log_probabilities(updated)

        start_counts = np.zeros(self.n_states)
        start_counts[self.start_mask] = start[self.start_mask] + TRANSITION_PSEUDOCOUNT
        self.log_start = _log_probabilities(start_counts / start_counts.sum())

    def _draw(self, log_probs: np.ndarray, rng: np.random.Generator) -> int:
        probabilities = np.exp(log_probs)
        total = float(probabilities.sum())
        if total <= 0.0:
            raise RuntimeError("Cannot draw a phase from an empty distribution.")
        return int(rng.choice(self.n_states, p=probabilities / total))


def merge_hmm(feature_names: Sequence[str]) -> GaussianHMM:
    """The six-phase merge model. Edges match the brief and figure 05."""
    names = tuple(feature_names)
    index = {name: i for i, name in enumerate(PHASES)}
    mask = np.zeros((len(PHASES), len(PHASES)), dtype=bool)
    for source, targets in DEFAULT_TRANSITIONS.items():
        for target in targets:
            mask[index[source], index[target]] = True
    start = np.asarray([name in DEFAULT_START for name in PHASES], dtype=bool)
    return GaussianHMM(
        state_names=PHASES,
        feature_names=names,
        transition_mask=mask,
        start_mask=start,
    )
