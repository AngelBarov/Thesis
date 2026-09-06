"""Players, actions, and parameters for the game-theoretic decision model.

Follows Section 3 of the supervisor model brief.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from .data import TrialData


class Player(str, Enum):
    """Vehicles in the merge scenario."""

    A = "A"
    L = "L"
    R = "R"


HUMAN_PLAYERS: tuple[Player, ...] = (Player.L, Player.R)


class Action(str, Enum):
    """Initial action sets.

    A is an observed first move unless the experiment contains several AV
    scripts, in which case its actions can be modelled explicitly.
    """

    SIGNAL = "Signal"
    DECELERATE = "Decelerate"
    MERGE = "Merge"

    YIELD = "Yield"
    CONTINUE = "Continue"

    BRAKE = "Brake"
    MAINTAIN = "Maintain"


ACTION_SETS: Mapping[Player, tuple[Action, ...]] = {
    Player.A: (Action.SIGNAL, Action.DECELERATE, Action.MERGE),
    Player.L: (Action.YIELD, Action.CONTINUE),
    Player.R: (Action.BRAKE, Action.MAINTAIN),
}


@dataclass(frozen=True)
class JointAction:
    """One combination of actions across the three vehicles.

    Unspecified players are ``None``, which is how the pairwise baseline games
    A--L and A--R ignore the third vehicle.
    """

    av: Action | None = None
    left: Action | None = None
    rear: Action | None = None

    def of(self, player: Player) -> Action | None:
        """Return the action assigned to ``player``."""
        return {Player.A: self.av, Player.L: self.left, Player.R: self.rear}[player]

    def with_action(self, player: Player, action: Action | None) -> JointAction:
        """Return a copy in which ``player`` takes ``action``."""
        if player is Player.A:
            return JointAction(av=action, left=self.left, rear=self.rear)
        if player is Player.L:
            return JointAction(av=self.av, left=action, rear=self.rear)
        return JointAction(av=self.av, left=self.left, rear=action)


@dataclass(frozen=True)
class CostWeights:
    """Component weights for one player.

    C_i(a) = w_s S_i(a) + w_e E_i(a) + w_c K_i(a) [+ w_I I_i(a)]

    The interaction weight stays ``None`` while the optional interaction cost
    is excluded.
    """

    safety: float
    efficiency: float
    comfort: float
    interaction: float | None = None

    @property
    def uses_interaction(self) -> bool:
        return self.interaction is not None

    def as_vector(self) -> tuple[float, ...]:
        """Weights in safety, efficiency, comfort[, interaction] order."""
        base = (self.safety, self.efficiency, self.comfort)
        if self.interaction is None:
            return base
        return (*base, self.interaction)


AlphaFn = Callable[[Player, "TrialData"], float]


@dataclass
class PlayerParameters:
    """Everything the decision rule needs for one player.

    ``alpha`` is the choice-consistency parameter of the probabilistic rule.
    Set ``alpha_fn`` instead to let it depend on human factors, as in
    alpha_i = f(stress, AV exposure, reaction-time history). A latent
    behavioural profile can supply either the weights or alpha.
    """

    weights: CostWeights
    alpha: float = 1.0
    alpha_fn: AlphaFn | None = None

    def choice_consistency(self, player: Player, data: "TrialData") -> float:
        """Resolve alpha_i for this trial."""
        if self.alpha_fn is None:
            return self.alpha
        return float(self.alpha_fn(player, data))
