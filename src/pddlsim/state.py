"""Items related to storing the state of a PDDLSIM simulation, in predicates."""

from collections.abc import Iterator, MutableSet
from dataclasses import dataclass, field
from random import Random

from pddlsim.ast import (
    AndCondition,
    AndEffect,
    Atom,
    Condition,
    Effect,
    EqualityCondition,
    NotCondition,
    NotPredicate,
    Object,
    OrCondition,
    Predicate,
    ProbabilisticEffect,
)


@dataclass(eq=True, frozen=True)
class SimulationState:
    """Data structure storing the environment state of a PDDLSIM simulation.

    As PDDLSIM is PDDL based, the state is a set of true grounded predicates.
    `SimulationState` is immutable, meaning methods that "modify" it, actually
    return a new state (e.g., `SimulationState.make_effect_hold`).

    Constructing a `SimulationState` is done by passing to the constructor
    a set of predicates, or calling it with no parameters, returning an
    empty state.
    """

    _true_predicates: MutableSet[Predicate[Object]] = field(default_factory=set)

    def _copy(self) -> "SimulationState":
        return SimulationState(set(self._true_predicates))

    def _does_atom_hold(self, atom: Atom[Object]) -> bool:
        match atom:
            case Predicate():
                return atom in self._true_predicates
            case NotPredicate(base_predicate):
                return base_predicate not in self._true_predicates

    def _make_atom_hold(self, atom: Atom[Object]) -> None:
        match atom:
            case Predicate():
                self._true_predicates.add(atom)
            case NotPredicate(base_predicate):
                self._true_predicates.discard(base_predicate)

    def does_condition_hold(self, condition: Condition[Object]) -> bool:
        """Check if the given grounded condition holds in the state."""
        match condition:
            case AndCondition(subconditions):
                return all(
                    self.does_condition_hold(subcondition)
                    for subcondition in subconditions
                )
            case OrCondition(subconditions):
                return any(
                    self.does_condition_hold(subcondition)
                    for subcondition in subconditions
                )
            case NotCondition(base_condition):
                return not (self.does_condition_hold(base_condition))
            case EqualityCondition(left_side, right_side):
                return left_side == right_side
            case Predicate():
                return self._does_atom_hold(condition)

    def _make_effect_hold(self, effect: Effect[Object], rng: Random) -> None:
        match effect:
            case AndEffect(subeffects):
                for subeffect in subeffects:
                    self._make_effect_hold(subeffect, rng)
            case ProbabilisticEffect():
                self._make_effect_hold(effect.choose_possibility(rng), rng)
            case Predicate() | NotPredicate():
                self._make_atom_hold(effect)

    def make_effect_hold(
        self, effect: Effect[Object], rng: Random | None = None
    ) -> "SimulationState":
        """Make the effect hold in a new returned state.

        Constructs a new state in which the given effect holds,
        while choosing subeffects in probabilistic junctions
        (`pddlsim.ast.ProbabilisticEffect`) randomly, and based
        on a `random.Random` object, if passed.
        """
        new_state = self._copy()

        new_state._make_effect_hold(effect, rng if rng else Random())

        return new_state

    def __iter__(self) -> Iterator[Predicate[Object]]:
        """Return an iterator over all grounded predicates in the state."""
        return iter(self._true_predicates)
