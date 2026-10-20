# Visualisation ideas for the thesis results

Eleven figure mock-ups for the three models in the supervisor brief: one shared
scenario figure, three per model, and one comparison figure.

Regenerate them with `python "Scripts/Visualization/generate_figures.py"`.
The scripts always write into this folder.

Slide pictures of how a decision is made, rather than result graphs, are in
`presentation/`. Regenerate those with
`python "Scripts/Visualization/presentation.py"`.

**All numbers are invented.** These are layouts, not results. Each figure carries
a grey footnote saying so. Sized for the brief's A4 text width (6.3 in).

---

## Shared

### 01 — Scenario geometry

![scenario](01_scenario_schematic.png)

Names every quantity the costs are built from. Forces the gap definitions to be
fixed before anything is computed.

---

## Model 1 — game-theoretic decision model

### 02 — Sequential structure

![game tree](02_game_tree.png)

Example tree for driver L, with every path from the HMM phase chain.
Approach leads to Anticipation, where L's situation becomes Yield, Continue,
or Conflict. Each of those three ends in Resolution. The probability is on
the branch and L's cost is in the box.

### Game tree, extra

![game tree extra](game_tree_extra.png)

The same chain for L, drawn out: where a trial can start, the chance of staying
in a phase, the per-step exit, what is observed, and a Resolution ending on
Yield, Continue, and Conflict. This file is the figure itself.

### 03 — Cost decomposition

![cost decomposition](03_cost_decomposition.png)

Each action's cost split into its weighted components, so the reader sees *why*
one action wins. The optional interaction term is hatched, and sits at zero for
R — how a dropped term should look.

### 04 — Decision map

![decision map](04_decision_map.png)

Predicted P(Yield) over two kinematic variables with the observed choices on
top. Agreement shows as circles clustering in the blue region.

---

## Model 2 — Hidden Markov Model

Figure 02 draws the same chain: Approach, Anticipation, then Yield, Continue,
or Conflict, and Resolution at the end of each.

### 05 — Phase transitions

![transitions](05_hmm_transitions.png)

Arrow width is the transition probability; the curved arrow is the probability
of staying in a phase.

### 06 — One trial

![single trial](06_single_trial_phases.png)

Phase shading behind the measured signals, aligned to the AV cue, with reaction
time marked. The figure that shows the phases correspond to something visible.

### 07 — Phases across trials

![raster](07_phase_raster.png)

One row per trial, grouped by participant. If anticipation shortens with
exposure, the yellow band narrows down each block.

---

## Model 3 — latent stress-adaptation model

### 08 — Profile flow

![profile flow](08_profile_flow.png)

Participants moving between profiles across exposures. Persistence shows as
straight bands, adaptation as crossing ribbons.

### 09 — What the profiles mean

![profile parameters](09_profile_parameters.png)

Cost weights and α per profile, next to the yielding curves they produce. The
right panel is the important half — parameter values alone say nothing.

### 10 — Reaction time

![reaction time](10_reaction_time.png)

Reaction time by chosen action, and across exposures with individual
participants faint behind the group mean.

---

## Comparison

### 11 — Model progression

![model comparison](11_model_comparison.png)

Baseline → pairwise game → full game → full game with profiles, scored on
held-out participants. Error bars must come from the participant splits.

---

## Notes

Build order if time is short: **03**, **04**, **08**, **11**. Figures **01** and
**02** can be finalised now, since they need no fitted results.

Precedents in `Papers/`: Mohammadi et al. (2025) Fig. 4 for figure 04;
Smirnov et al. (2021) Fig. 1 and Yao & Du (2022) Fig. 2 for figures 01–02;
Wei et al. (2013) Fig. 8 for figure 06; Stange, Kühn et al. (2022) Figs. 4–8 for
figure 10. Two departures worth knowing: none of these papers plots individual
participants (figure 10 does) or reaction-time distributions, and Stange's
"repeated interactions" are condition levels rather than repeated trials.
