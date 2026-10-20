"""Phase set for the reference HMM.

Names and allowed moves follow the brief and figure 05. The probabilities
on those edges are estimated later; nothing in this file is a fitted number.
"Continue" is the same response the game calls Continue.
"""

from __future__ import annotations

PHASES: tuple[str, ...] = (
    "Approach",
    "Anticipation",
    "Yield",
    "Continue",
    "Conflict",
    "Resolution",
)

# Yield, Continue, and Conflict are alternative ways out of Anticipation,
# not a sequence. Resolution stays absorbing so a finished merge does not
# reopen. Backward edges are excluded on purpose: short trials cannot
# identify a free 6-by-6 matrix.
DEFAULT_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "Approach": ("Approach", "Anticipation"),
    "Anticipation": ("Anticipation", "Yield", "Continue", "Conflict"),
    "Yield": ("Yield", "Resolution"),
    "Continue": ("Continue", "Resolution"),
    "Conflict": ("Conflict", "Resolution"),
    "Resolution": ("Resolution",),
}

# A window that starts at the cue may already be past Approach.
DEFAULT_START: tuple[str, ...] = ("Approach", "Anticipation")
