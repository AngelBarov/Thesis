"""Reference HMM for behavioural phases within one merge.

Not the decision model. The game explains the human response; this model
asks which phase the interaction is in. See the folder README.
"""

from .exposure import ExposureSplit, split_decoded, split_transitions
from .gaussian_hmm import GaussianHMM, merge_hmm
from .phases import DEFAULT_START, DEFAULT_TRANSITIONS, PHASES
from .sequences import (
    DEFAULT_FEATURES,
    OPTIONAL_FEATURES,
    DecodedTrial,
    Frame,
    TrialSequence,
    decode_trial,
    decode_trials,
    first_entry,
    fit_trials,
    occupancy_seconds,
)

__all__ = [
    "DEFAULT_FEATURES",
    "DEFAULT_START",
    "DEFAULT_TRANSITIONS",
    "OPTIONAL_FEATURES",
    "PHASES",
    "DecodedTrial",
    "ExposureSplit",
    "Frame",
    "GaussianHMM",
    "TrialSequence",
    "decode_trial",
    "decode_trials",
    "first_entry",
    "fit_trials",
    "merge_hmm",
    "occupancy_seconds",
    "split_decoded",
    "split_transitions",
]
