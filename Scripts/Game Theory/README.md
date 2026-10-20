# Cost terms

`costs.py` evaluates the weighted sum. `cost_terms.py` holds five formulas for each component. `compose()` picks one of each, and L and R can be given different results.

```python
from cost_terms import compose

terms = compose(
    safety="inverse_ttc",
    efficiency="squared_speed_loss",
    comfort="abs_acceleration",
    interaction="insufficient_gap",  # omit this argument to leave the term out
)
```

The defaults are the three example formulas in the brief.

## Inputs

Action-conditional numbers live at `extra["projected"]["L"]["Yield"]["ttc"]` (player and action are the enum values). The keys are `PROJECTED_KEYS`. `desired_speed` stays on the vehicle. The formulas do not project a trajectory: fill the dict from the kinematics once they exist. A missing input raises `MissingCostFeature`.

`gap` is this driver's own conflict gap. `offered_gap` is the gap left for the merging AV. Keeping them apart stops an interaction term from repeating the safety term.

Pass a large TTC or headway when the gap is opening. A negative value is rejected.

## Safety

| Name | Formula | Reads |
|---|---|---|
| `inverse_ttc` | 1 / max(TTC, ε) | `ttc` |
| `inverse_gap` | 1 / max(gap, ε) | `gap` |
| `required_braking` | max(0, required braking) | `required_braking` |
| `inverse_time_headway` | 1 / max(headway, ε) | `time_headway` |
| `exponential_ttc` | exp(−TTC / τ) | `ttc` |

The first four are the brief's example and the three replacements it names (projected minimum gap, time headway, required braking). `exponential_ttc` uses the same TTC but stays in (0, 1], so one very small TTC cannot swamp the sum.

## Efficiency

| Name | Formula | Reads |
|---|---|---|
| `squared_speed_loss` | (v_desired − v_after)² | `speed_after`, desired speed |
| `speed_loss` | max(0, v_desired − v_after) | same |
| `time_delay` | extra seconds over a reference distance | same, `reference_distance_m` |
| `time_below_desired` | seconds spent below desired speed | `time_below_desired` |
| `distance_lost` | metres of progress given up | `distance_lost` |

`squared_speed_loss` is the brief's equation, so a speed gain is charged as well. `speed_loss` charges only slowing down. Delay, time below desired speed, and distance lost are the other definitions the brief names. The last two are integrals supplied by the projection.

## Comfort

| Name | Formula | Reads |
|---|---|---|
| `abs_acceleration` | \|a\| | `acceleration` |
| `squared_acceleration` | a² | `acceleration` |
| `abs_jerk` | \|jerk\| | `jerk` |
| `squared_jerk` | jerk² | `jerk` |
| `harsh_acceleration` | max(0, \|a\| − a_comfort) | `acceleration` |

`abs_acceleration` is the brief's example. The jerk pair is there for when that derivative is usable. `harsh_acceleration` is zero inside a comfort band.

## Interaction

Optional. Use one only when it measures something safety and efficiency do not, which is the brief's condition for keeping the term.

| Name | Formula | Reads |
|---|---|---|
| `insufficient_gap` | 0 if the offered gap is large enough, else 1 | `offered_gap`, `sufficient_gap_m` |
| `gap_shortfall` | max(0, g\* − offered gap) | same |
| `gap_closure` | max(0, −gap change) | `gap_change` |
| `imposed_braking` | max(0, braking forced on another vehicle) | `imposed_braking` |
| `inverse_offered_gap` | 1 / max(offered gap, ε) | `offered_gap` |

`insufficient_gap` is the brief's example. Set `sufficient_gap_m` on the trial; there is no default gap.

## Constants that the weight does not replace

ε is 0.1 s, 0.1 m, or 0.1 m/s. τ defaults to 1.5 s (`ttc_scale_s`) and the comfort band to 1.5 m/s² (`comfort_accel_mps2`). Those two sit inside the formula, so fix them before fitting rather than moving them to chase a hit rate. The reference distance defaults to 20 m (`reference_distance_m`); that scale is absorbed by the efficiency weight.

Compare remixes by held-out likelihood. The raw weights are not on a common scale, so a large efficiency weight only means that term was measured in smaller units.

`python "Scripts/Game Theory/example_stub.py"` checks the brief formulas on invented numbers.
