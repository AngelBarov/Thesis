# Agent Context: Models for AV Mandatory-Merge Thesis

> **Use this file** to give another AI agent compact context on modelling choices.  
> **Student:** Angel Barov | **Supervisor:** Tianyu Tang | **Thesis:** game-theoretic modelling of **human driver adaptation** during AV mandatory merge.

---

## Task (one paragraph)

Controlled driving experiment: **AV (A)** in ending lane must merge **left**; **human L** in target lane; **human R** behind A. **Do not design the AV controller.** Predict how **L and R adapt** (yield/hold, brake/maintain, timing) when A cues merge. Compare candidate models; pick best-supported formulation. Model **human adaptation**, not scenario replay. Multiple experimental **scenarios** may exist — analyse only comparable ones.

**Targets to predict:** L yield vs hold; R brake vs maintain; reaction time; optional participant/repetition effects.

---

## Scenario entities

| ID | Role | First action set |
|----|------|------------------|
| A | AV (observed cue / fixed script) | indicator, deceleration, lateral move |
| L | left-lane human | Yield, Hold |
| R | rear human | Brake, Maintain |

Expand later if data support: L → Maintain/Assert; R → Accelerate.

---

## Model candidates (compare these)

### M1 — Cost-based sequential response game ⭐ START HERE

- **Type:** Sequential game + estimated cost functions + quantal response
- **Structure:** `A cue → L chooses → R chooses` (or parallel L/R if uncoupled)
- **Cost terms:** safety (TTC⁻¹ or min gap), progress (speed loss), comfort (|accel|), interaction (gap sufficiency)
- **Decision rule:** softmax over costs; weights `w` estimated from data, not hand-tuned
- **When to use:** Default first model; supervisor-aligned; interpretable; fits mandatory merge
- **Refs:** Chen et al. (2023); Mohammadi et al. (2025); Ji & Levinson (2020)
- **Motivation:** Humans trade safety vs progress; costs explain Yield/Hold and Brake/Maintain without assuming full rationality

### M2 — Coupled L–R game

- **Type:** Same as M1 but `C_L(a_L, a_R)` and `C_R(a_R, a_L)` with expected opponent action
- **When to use:** Only if L and R can observe or influence each other in data
- **Refs:** Mohammadi et al. (2025); Levinson (2005)
- **Motivation:** Rear braking may depend on whether L opens gap; strategic interdependence

### M3 — Incomplete-information (Bayesian) game

- **Type:** Hidden driver type θ (cooperative / neutral / aggressive) → different cost weights or priors
- **When to use:** Strong participant heterogeneity; emotion/style data available
- **Refs:** Yao & Du (2022); Li et al. (2026)
- **Motivation:** L and R do not know each other's cooperativeness; type explains variable yielding

### M4 — Stackelberg lane-change game

- **Type:** Leader (merger) moves first; follower reacts; safety + space payoffs
- **When to use:** If A's merge timing is strategic variable; closest published lane-change template
- **Refs:** Smirnov et al. (2021); Ji & Levinson (2020)
- **Motivation:** Standard game template for lane change; adapt players to A, L, R

### M5 — Static Nash game

- **Type:** Simultaneous discrete actions; Nash equilibrium
- **When to use:** Simple baseline; likely too rigid for sequential merge
- **Refs:** Ji & Levinson (2020); Levinson (2005)
- **Motivation:** Minimal strategic baseline for comparison

### M6 — Participant + repetition extensions (on top of M1–M3)

- **Type:** Per-participant weights `w_{h,p}`; trial effect `w(e) = w₀ + δ·e`
- **When to use:** After base model works; test adaptation over repeated AV exposure
- **Refs:** Stange et al. (2022); Li et al. (2026)
- **Motivation:** Humans change headway/speed with repeated AV encounters; test if cost weights shift

---

## Non-game reference models (benchmarks)

### B1 — Descriptive statistical baseline

- **Type:** Logistic/regression on gaps, rel speed, TTC, distance to lane end
- **When to use:** Required benchmark; shows if game structure adds value
- **Motivation:** Minimal predictor without strategic structure

### B2 — Hidden Markov Model (HMM)

- **Type:** Latent behavioural phases from trajectories (approach, negotiate, yield, merge)
- **When to use:** Optional; **not main thesis framework**; sequential pattern check only
- **Refs:** Rabiner (1989); Tang (2026) for action-chain framing
- **Motivation:** Reveals phase sequences game model might miss; no explicit payoffs/beliefs

---

## Conceptual / motivation refs (not primary estimators)

| Ref | Why cite |
|-----|----------|
| Markkula et al. (2020) | Define interaction episodes; space-sharing conflict |
| Tang (2026) | Action chains → scripts → interaction patterns |
| Wei et al. (2013) | Social merge + intention uncertainty; roles reversed vs our setup |
| Schwarting et al. (2018) | Behaviour-aware planning context |

---

## Evaluation protocol

1. Segment trials: AV cue → human decision window → action → outcome
2. Fit on training participants/trials
3. **Hold out entire participants** for generalisation test
4. Compare: B1 → M1 → M2/M3/M6 as justified
5. Metrics: action classification accuracy, calibration, reaction-time error, lift over B1

**AV cue candidates:** indicator onset | first sustained deceleration | first lateral movement (confirm with supervisor)

---

## Key references (APA-style + DOI)

```
Markkula, G., et al. (2020). Defining interactions. TIES, 21(6), 728–752.
  https://doi.org/10.1080/1463922X.2020.1736686

Ji, A., & Levinson, D. (2020). Game theory models of lane changing. Transportmetrica A, 16(3), 1628–1647.
  https://doi.org/10.1080/23249935.2020.1793370

Chen, Y., et al. (2023). Mandatory merging game model. Mathematics, 11(2), 402.
  https://doi.org/10.3390/math11020402

Smirnov, V., et al. (2021). Game theoretic lane changing. Sensors, 21(5), 1523.
  https://doi.org/10.3390/s21051523

Yao, R., & Du, X. (2022). Lane changing + driving styles game. Travel Behaviour and Society, 29, 319–329.
  https://doi.org/10.1016/j.tbs.2022.07.008

Mohammadi, A., et al. (2025). Behavioral game, coupled decisions. TRF, 112, 48–62.
  https://doi.org/10.1016/j.trf.2025.03.026

Stange, V., et al. (2022). Repeated interactions with L3 AVs. TRF, 87, 426–443.
  https://doi.org/10.1016/j.trf.2022.04.019

Li, R., et al. (2026). Evolutionary game, AV social preferences. AAP, 228, 108402.
  https://doi.org/10.1016/j.aap.2026.108402

Rabiner, L. R. (1989). HMM tutorial. Proc. IEEE, 77(2), 257–286.
  https://doi.org/10.1109/5.18626
```

---

## Constraints (do not violate)

- Game theory is **central**; HMM is **reference only**
- Model **human adaptation**, not mimic full driving scenario
- Term **scenario** = experimental design condition only
- Start **simple** (2 actions per human); add coupling/types/repetition only if data support
- Weights and parameters **estimated from data**, not set to desired outcomes

---

## Related files in repo

- `Scripts/Model Brief/supervisor_model_brief.md` — full maths and cost-function detail
- `Scripts/Model Brief/supervisor_model_brief.tex` / `.pdf` — compiled brief for supervisor
- `Documents/Expose.docx` — exposé draft
- `Documents/Report until now.txt` — progress notes and open questions
