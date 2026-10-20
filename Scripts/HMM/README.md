# Hidden Markov model

Reference model from the brief (Model 2). It asks which behavioural phase a merge is in. The game remains the model of the decision.

One sequence per trial, shared by A, L, and R. Each phase emits a diagonal Gaussian: one mean and one variance per feature. That is Rabiner's forward, Viterbi, and Baum–Welch calculation with a continuous emission, so speed and gap do not have to be binned into symbols.

## Phases

Approach, then Anticipation, then one of Yield, Continue, or Conflict, then Resolution.

| Phase | Allowed next phase |
|---|---|
| Approach | Approach, Anticipation |
| Anticipation | Anticipation, Yield, Continue, Conflict |
| Yield | Yield, Resolution |
| Continue | Continue, Resolution |
| Conflict | Conflict, Resolution |
| Resolution | Resolution |

A trial may start in Approach or Anticipation. The mask is the chain in the brief and in figure 05. `fit` estimates the probabilities on the allowed edges only.

Yield, Continue, and Conflict have the same allowed moves, so those three labels can swap during fitting. Read `emission_means()` before using the names. Approach and Resolution are tied down by the mask and by the initialisation, which seeds the first phase from early frames and the last phase from late frames.

## Frames

`TrialSequence` is the time series. The game's `TrialData` stays a single decision instant.

Put time in seconds relative to the AV cue, on one shared step stored as `frame_dt`. A transition probability is per step, so every trial uses the same step. Resample before building the sequence. The model does not fill missing channels.

`DEFAULT_FEATURES`: speed and acceleration of L and R, gap ahead of L, L's relative speed to A, gap from R to A, AV indicator, AV lateral position.

`OPTIONAL_FEATURES`: TTC, A's speed and acceleration, the human action, stress.

- TTC repeats gap and relative speed, so it stays off unless a kinematic channel is missing.
- The action happens once. Put it only on frames where the driver has already shown it; copying the final choice onto the whole trial lets the phase model see the decision early.
- Stress is included only when it is recorded on the kinematic clock. As a predictor of the decision, it belongs to the profile model.

Recommended sign when the columns are built: positive relative speed means the follower is closing.

## What to call once the sequences exist

1. Hold out whole participants, same split as the game.
2. `fit_trials` on the training sequences. The object keeps that sample's mean and variance, and later trials are scaled with it.
3. `decode_trials` for the phase path in figures 06 and 07. `first_entry` and `occupancy_seconds` measure how long a phase lasts.
4. `log_likelihood` on held-out trials when comparing models. The score is for the sequence, not for Yield versus Continue.
5. `split_decoded` counts transitions in early exposures and in later ones. The count is descriptive. Exposure is not an input of the transition matrix.

`python "Scripts/HMM/example.py"` checks the forward pass and Viterbi against every path of a small model, recovers a synthetic 3-phase chain, and checks the merge mask. It does not read experimental data.

Numpy is the only added dependency.

## Decide with the logs, not before

Pick the frame step from the recording rate. 10 Hz is a workable start when the logs are at least that fast. Drop a default feature that is missing or redundant. Widen `DEFAULT_TRANSITIONS` if the decoded paths pile up on an edge the mask forbids.
