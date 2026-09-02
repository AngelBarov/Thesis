# Game-Theoretic Modelling of Human Driver Responses During an AV Merge

**Student:** Angel Barov  
**Supervisor:** Tianyu Tang  
**Purpose:** Simple modelling proposal for discussion

> For correctly rendered formulas, use `supervisor_model_brief.pdf`.  
> The Markdown file requires a preview extension that supports `$` and `$$` mathematics.
>
> To change the final PDF, edit `supervisor_model_brief.tex`, save it, then double-click `build_supervisor_brief.bat`.

## 1. What the thesis is trying to predict

The experiment contains three vehicles:

- **A:** the automated vehicle in the ending lane;
- **L:** the human-driven vehicle in the left target lane;
- **R:** the human-driven vehicle behind A.

The thesis focuses on the decisions of the two human drivers. It does not try to design the AV controller.

The main predictions are:

- Will L yield or hold position?
- Will R brake or maintain speed?
- When will each human react?
- Do these decisions change between drivers or after repeated trials?

## 2. Where the game theory enters

A game consists of four elements:

1. **Players:** the decision makers;
2. **Actions:** the choices available to each player;
3. **Costs:** how undesirable each possible result is for each player;
4. **Decision rule:** the action expected from the costs.

The proposed model is a **sequential response game**:

1. The AV gives an observable cue, such as indicating, decelerating, or moving laterally.
2. L and R observe the situation.
3. Each human chooses a response.
4. The cost of each response depends on safety, progress, comfort, and the expected behavior of the other vehicles.

The game can be represented as:

$$
\text{AV action} \longrightarrow
\begin{cases}
\text{L chooses yield or hold},\\
\text{R chooses brake or maintain}.
\end{cases}
$$

This is game-theoretic because the best action for one driver depends on what the other vehicles are doing or are expected to do. For example, holding position may be attractive when the AV is far away, but costly when the AV is already entering the gap.

If the AV always follows one fixed script, its behavior is treated as an observed first move. The thesis then estimates the human response part of the game. If several AV strategies exist, they can be compared explicitly.

## 3. Players and actions

For a first simple version:

| Player | Actions |
|---|---|
| AV, A | observed merge cue or merge behavior |
| Left-lane human, L | Yield, Hold |
| Rear human, R | Brake, Maintain |

The action sets can later be expanded:

- L: Yield, Maintain, Assert;
- R: Brake, Maintain, Accelerate.

Starting with two actions per human makes the model easier to estimate and explain.

## 4. A simple cost function

For human driver $h$, define the cost of action $a$ as:

$$
C_h(a)
=
w_{s,h}R_h(a)
+
w_{p,h}P_h(a)
+
w_{c,h}K_h(a)
+
w_{i,h}I_h(a)
$$

The terms mean:

- $R_h(a)$: **safety risk** after taking action $a$;
- $P_h(a)$: **progress cost**, such as losing speed or time;
- $K_h(a)$: **comfort cost**, such as strong braking or jerk;
- $I_h(a)$: **interaction cost**, such as blocking an expected merge or creating an insufficient gap;
- $w_{s,h}, w_{p,h}, w_{c,h}, w_{i,h}$: estimated importance weights.

A lower cost means that the action is more attractive.

The weights are not chosen manually to produce a desired result. They are estimated from the observed human decisions.

### 4.1 Safety risk

A simple safety measure is inverse time to collision:

$$
R_h(a)=\frac{1}{\max(\mathrm{TTC}_h(a),\varepsilon)}
$$

Here:

- $\mathrm{TTC}_h(a)$ is the projected time to collision if the human takes action $a$;
- $\varepsilon$ is a small positive value that prevents division by zero.

Small TTC gives a large safety cost. Large TTC gives a small safety cost.

If TTC is not appropriate in a particular vehicle arrangement, required braking or projected minimum gap can be used instead.

### 4.2 Progress cost

The progress cost measures how much speed the driver gives up:

$$
P_h(a)=\left(v_h^{\mathrm{desired}}-v_h^{\mathrm{after}}(a)\right)^2
$$

Yielding or braking may improve safety but increase progress cost.

### 4.3 Comfort cost

A simple comfort cost is:

$$
K_h(a)=|a_h(a)|
$$

where $a_h(a)$ is the acceleration required by the action. Strong acceleration or braking produces a larger cost.

Jerk can be added later if the acceleration measurements are reliable.

### 4.4 Interaction cost

The interaction term describes whether an action makes the merge harder or less predictable:

$$
I_L(a)=
\begin{cases}
0, & \text{if L creates a sufficient gap},\\
1, & \text{if L leaves an insufficient gap}.
\end{cases}
$$

This term should be defined from observable gap changes. It should not be treated as a direct measurement of kindness or intention.

## 5. How the model predicts a decision

### 5.1 Simplest decision rule

The simplest model predicts that a human chooses the action with the lowest cost:

$$
a_h^*=\underset{a}{\operatorname{argmin}}\;C_h(a)
$$

For L:

$$
\text{Predict Yield if }
C_L(\mathrm{Yield}) < C_L(\mathrm{Hold})
$$

For R:

$$
\text{Predict Brake if }
C_R(\mathrm{Brake}) < C_R(\mathrm{Maintain})
$$

This is a deterministic best-response model.

### 5.2 More realistic probabilistic decision rule

Humans do not always choose the mathematically lowest-cost action. The model can therefore convert costs into probabilities:

$$
P_h(a)=
\frac{\exp(-\lambda_h C_h(a))}
{\sum_b \exp(-\lambda_h C_h(b))}
$$

The parameter $\lambda_h$ describes choice consistency:

- small $\lambda_h$: decisions are more variable;
- large $\lambda_h$: the lowest-cost action is chosen more consistently.

This is the bounded-rational or quantal-response part of the model.

## 6. Strategic dependence between L and R

The first model can treat L and R as separate responses to the AV.

The game-theoretic extension allows the cost for one human to depend on the expected action of the other human:

$$
C_L(a_L,a_R)
$$

and

$$
C_R(a_R,a_L)
$$

For example, L opening a gap may let the AV merge with less braking. This can reduce the following risk for R. The joint result therefore depends on both human actions.

Because a driver may not know the other human's next action, use expected cost:

$$
EC_L(a_L)
=
\sum_{a_R}
P(a_R\mid\text{information observed by L})\,C_L(a_L,a_R)
$$

$$
EC_R(a_R)
=
\sum_{a_L}
P(a_L\mid\text{information observed by R})\,C_R(a_R,a_L)
$$

In words, the driver considers each possible action of the other human, weights it by how likely it seems, and chooses the response with the smallest expected cost.

This coupled model should only be used if L and R can observe or physically influence one another. Otherwise, they may simply be two independent humans responding to the same AV.

## 7. Example of one decision

Suppose L must choose Yield or Hold.

The model calculates:

$$
C_L(\mathrm{Yield})
=
w_sR_{\mathrm{Yield}}
+
w_pP_{\mathrm{Yield}}
+
w_cK_{\mathrm{Yield}}
+
w_iI_{\mathrm{Yield}}
$$

$$
C_L(\mathrm{Hold})
=
w_sR_{\mathrm{Hold}}
+
w_pP_{\mathrm{Hold}}
+
w_cK_{\mathrm{Hold}}
+
w_iI_{\mathrm{Hold}}
$$

Typical trade-off:

- Yield has lower safety and interaction cost;
- Yield may have higher progress and comfort cost;
- Hold preserves speed but may create higher risk.

The estimated weights determine which trade-off best explains the observed human choice.

The same structure is used for R with Brake and Maintain.

## 8. Driver differences and repeated exposure

Different drivers may value the cost terms differently:

$$
C_{h,p}(a)
=
w_{s,h,p}R_h(a)
+
w_{p,h,p}P_h(a)
+
w_{c,h,p}K_h(a)
+
w_{i,h,p}I_h(a)
$$

Here $p$ identifies the participant.

Repeated exposure can be tested by allowing one or more weights to change with trial number. For example:

$$
w_{s,h,p}(e)=w_{s,h,p}^{0}+\delta_h e
$$

where:

- $e$ is the repetition number;
- $w^0$ is the initial safety weight;
- $\delta_h$ is the estimated change across repetitions.

If $\delta_h$ is positive, safety becomes more influential over repeated trials. If it is negative, drivers may become more willing to accept smaller safety margins. The direction must be learned from the data.

## 9. Reaction time

Reaction time can be defined simply as:

$$
\mathrm{RT}_h
=
t_{\mathrm{human\ response}}
-
t_{\mathrm{AV\ cue}}
$$

Possible AV cue times:

- indicator onset;
- first sustained deceleration;
- first visible lateral movement.

The first analysis can compare average reaction time by action, participant, and repetition. A more advanced timing model can be added only if needed.

## 10. Data needed to calculate the costs

The main measured or derived variables are:

- position and speed of all vehicles;
- acceleration;
- bumper-to-bumper gaps;
- relative speed;
- TTC or required braking;
- AV indicator state;
- AV lateral movement;
- trial number;
- participant identity;
- stress and gaze, if synchronized and reliable.

The data must be segmented into:

1. AV cue;
2. human decision window;
3. observed human action;
4. outcome and resolution.

## 11. Estimating and testing the model

The unknown cost weights are fitted so that the predicted decisions match the observed human decisions.

The main comparison should be:

1. a simple statistical model using gaps and speeds;
2. the cost-based game model;
3. the cost-based model with driver differences;
4. the cost-based model with repeated-exposure effects.

Testing must leave complete participants out of the training data. This shows whether the model can predict a new human driver instead of remembering one participant's behavior.

Useful evaluation measures:

- percentage of correctly predicted Yield, Hold, Brake, and Maintain decisions;
- probability calibration;
- prediction error for reaction time;
- improvement over the simple statistical baseline.

## 12. Recommended first model

The recommended first game-theoretic model is:

$$
\boxed{
\text{Observed AV action}
\rightarrow
\text{human cost calculation}
\rightarrow
\text{probability of each human response}
}
$$

In practical terms:

- estimate separate cost functions for L and R;
- start with two actions per human;
- use safety, progress, comfort, and interaction costs;
- estimate the weights from observed decisions;
- add coupling between L and R only if the data support it;
- add participant and repetition effects after the basic model works.

## 13. Other useful models to compare

The cost-based sequential game should remain the main model. The following models answer different questions and can be added only when they are supported by the data.

### 13.1 Incomplete-information game

Drivers do not know exactly how cooperative, cautious, or assertive another driver is. An incomplete-information game represents this uncertainty using hidden **driver types**.

For example, L may initially believe that the AV is cautious or assertive. As the AV signals, decelerates, and moves laterally, L updates this belief and chooses Yield or Hold.

Why it may be useful:

- it represents uncertainty about another road user's behavior;
- it can explain why two drivers react differently to the same AV action;
- it allows beliefs to change as more behavior is observed;
- stress or gaze could help describe a type, but only as an exploratory extension.

Main limitation:

- several repeated observations per participant are needed to distinguish a stable type from random behavior.

### 13.2 Behavioral evidence-accumulation game

A normal best-response model assumes that a decision is made immediately from the current costs. A behavioral evidence-accumulation model instead assumes that preference builds over time.

For example, L may not yield as soon as the indicator appears. Continued AV deceleration and lateral movement provide additional evidence until Yield becomes the preferred action.

Why it may be useful:

- it represents negotiation as a process rather than one instant;
- it can explain delayed or changing decisions;
- it connects visible AV cues to reaction time;
- it allows humans to behave imperfectly rather than as perfectly rational players.

Main limitation:

- it requires reliable high-frequency data and a defensible definition of when evidence accumulation starts.

### 13.3 Risk-threshold model

This model assumes that a driver keeps the current plan until perceived risk exceeds a personal threshold:

$$
\text{Change action if perceived risk}>\text{driver threshold}
$$

A cautious driver may have a low threshold and react early. A more risk-tolerant driver may wait longer.

Why it may be useful:

- it gives a simple explanation for reaction timing;
- it represents individual differences clearly;
- it uses continuous kinematic communication such as gaps, acceleration, and lateral movement;
- it does not assume that humans continuously calculate an optimal solution.

Main limitation:

- perceived risk and the threshold are inferred from behavior, not measured directly.

### 13.4 Hidden Markov Model

An HMM can describe recurring hidden phases such as:

$$
\text{Approach}
\rightarrow
\text{Anticipation}
\rightarrow
\text{Yield or Conflict}
\rightarrow
\text{Resolution}
$$

Why it may be useful:

- it can segment long trajectories into interpretable phases;
- it can reveal common action sequences;
- it provides a non-game-theoretic comparison;
- it may show whether repeated exposure changes transitions between phases.

Main limitation:

- an HMM describes temporal patterns but does not explain strategic dependence between road users;
- a hidden state is a statistical pattern, not direct evidence of intention.

### 13.5 Full three-player game

The complete model could treat A, L, and R as three strategic players, each with actions and costs.

Why it may be useful:

- it represents all three vehicles in one joint model;
- it can capture how L's response affects the AV and then affects R;
- it is closest to the complete experimental situation.

Main limitation:

- it requires multiple AV strategies and evidence that all players adapt to one another;
- it introduces many costs and parameters, which may not be identifiable from a small dataset;
- if the AV follows one fixed script, treating it as a strategic player would be misleading.

### 13.6 Latent stress-adaptation profile model

This is the model closest to what you describe. It is related to the incomplete-information game and the HMM, but it is not identical to either:

- the incomplete-information game usually assumes that a driver's hidden type is relatively stable;
- the earlier HMM describes short phases within one merge;
- this new model allows a participant's broader behavioral profile to change between repeated trials.

At each exposure, a participant belongs probabilistically to an unobserved profile:

$$
z_{p,e}\in
\{\text{Profile 1},\text{Profile 2},\text{Profile 3}\}
$$

The profiles should initially have neutral names. After fitting the model, their behavior may support descriptions such as:

- **cautious or stressed:** earlier braking, more yielding, larger gaps;
- **balanced or adaptive:** moderate responses that depend on context;
- **assertive or habituated:** later reactions, less yielding, smaller accepted gaps.

These labels describe observed patterns. They should not be presented as fixed personality traits.

The probability of moving to a new profile depends on the previous profile and abstract human factors:

$$
P(\text{current profile})
=
f(
\text{previous profile},
\text{stress},
\text{exposure number},
\text{previous outcome}
)
$$

Possible inputs:

- stress level before the human decision;
- change in stress from the participant's normal baseline;
- number of previous AV encounters;
- outcome of the previous encounter;
- previous Yield, Hold, Brake, or Maintain response;
- gaze or attention measures;
- participant characteristics, if available and ethically appropriate.

Why it may be useful:

- it directly tests whether repeated AV exposure is associated with changing behavioral profiles;
- it distinguishes temporary stress from a more stable participant tendency;
- it uses human and psychological information rather than only gaps and speeds;
- it may identify participants who become more comfortable, more cautious, or more assertive over time;
- it provides an abstract baseline for judging how much physical traffic variables add.

The most informative comparison would be:

1. **Abstract baseline:** stress, exposure, previous outcome, and participant history only;
2. **Physical model:** gaps, relative speed, TTC, and acceleration only;
3. **Combined model:** abstract profile plus the physical cost-based game.

The combined game can allow the behavioral profile to change the cost weights. For example, a cautious profile may place more weight on safety, while an assertive profile may place more weight on maintaining speed.

Main limitations:

- stress must be measured before the predicted decision, otherwise it may be a consequence of the event rather than a predictor;
- enough repeated trials are needed to estimate transitions between profiles;
- profile labels must be based on model results, not decided in advance;
- this should be called a latent behavioral profile model, not proof that a person's real personality has changed.

This is a proposed thesis model built from the ideas of hidden driver types, repeated exposure, and latent-state modelling. None of the cited papers validates this exact combination for the present three-vehicle merge.

### 13.7 Suggested comparison order

| Priority | Model | Main question |
|---|---|---|
| 1 | Physical statistical baseline | Can gaps and speeds predict the response? |
| 2 | Abstract profile baseline | Can stress and exposure predict the response? |
| 3 | Cost-based sequential game | Which human trade-offs explain the response? |
| 4 | Combined profile and cost game | Do changing profiles improve the physical model? |
| 5 | Risk threshold or evidence accumulation | When and how does the response develop? |
| 6 | Incomplete-information game | Does uncertainty about driver type improve prediction? |
| 7 | HMM | Are there recurring temporal phases? |
| Optional | Full three-player game | Is full strategic coupling supported by the data? |

The best thesis model is not necessarily the most complex model. It is the simplest model that explains human behavior and improves prediction on unseen participants.

## 14. Questions for the supervisor

1. Does the AV use multiple merge strategies, or one fixed script?
2. Can L and R observe each other's behavior?
3. Are Yield/Hold and Brake/Maintain suitable first action sets?
4. Which AV cue should start the decision window?
5. Is TTC valid for every vehicle arrangement, or should projected gap and required braking be primary?
6. How many repeated trials are available per participant?
7. Should the first analysis model the final action only, or several decisions over time?
8. Is stress available continuously, and can a pre-decision stress value be calculated reliably?

## 15. Main motivating papers

- Markkula et al. (2020), definition of interaction:  
  https://doi.org/10.1080/1463922X.2020.1736686

- Ji and Levinson (2020), review of game-theoretic lane-changing models:  
  https://doi.org/10.1080/23249935.2020.1770368

- Chen et al. (2023), costs and bounded rationality in mandatory merging:  
  https://doi.org/10.3390/math11020402

- Yao and Du (2022), incomplete-information game and driver types:  
  https://doi.org/10.1016/j.tbs.2022.07.008

- Mohammadi et al. (2025), behavioral game with coupled decisions:  
  https://doi.org/10.1016/j.trf.2025.03.026

- Siebinga et al. (2024), communication and risk-threshold model:  
  https://doi.org/10.1093/pnasnexus/pgae420

- Rabiner (1989), Hidden Markov Model reference:  
  https://doi.org/10.1109/5.18626

- Stange et al. (2022), repeated exposure to automated vehicles:  
  https://doi.org/10.1016/j.trf.2022.04.019
