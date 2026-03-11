"""Items directly related to simulation, and low-level simulation APIs.

> [!TIP]
> This is a low-level API. If you want to interface with agents, `pddlsim.local`
> or `pddlsim.remote.server` may be a better fit.
"""

import os
from collections import defaultdict
from collections.abc import (
    Generator,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from functools import cached_property
import itertools
from random import Random
from typing import cast

from clingo import Control

from pddlsim._asp import (
    ASPPart,
    ASPPartKind,
    IDAllocator,
    ObjectNameID,
    PredicateID,
    TypeNameID,
    VariableID,
    action_definition_asp_part,
    objects_asp_part,
    simulation_state_asp_part,
)
from pddlsim.ast import (
    ActionDefinition,
    ActionFallibility,
    AndCondition,
    AndEffect,
    Argument,
    Condition,
    Domain,
    Effect,
    EqualityCondition,
    ForallCondition,
    GroundedAction,
    Identifier,
    NotCondition,
    NotPredicate,
    Object,
    OrCondition,
    Predicate,
    ProbabilisticEffect,
    Problem,
    Revealable,
    Type,
    Variable,
)
from pddlsim.state import SimulationState


def _ground_argument(
    argument: Argument, grounding: Mapping[Variable, Object]
) -> Object:
    return grounding[argument] if isinstance(argument, Variable) else argument


def _ground_predicate(
    predicate: Predicate[Argument], grounding: Mapping[Variable, Object]
) -> Predicate[Object]:
    return Predicate(
        predicate.name,
        tuple(
            _ground_argument(argument, grounding)
            for argument in predicate.assignment
        ),
    )


def _ground_condition(
    condition: Condition[Argument], grounding: Mapping[Variable, Object], problem: Problem
) -> Condition[Object]:
    match condition:
        case AndCondition(subconditions):
            return AndCondition(
                [
                    _ground_condition(subcondition, grounding, problem)
                    for subcondition in subconditions
                ]
            )
        case OrCondition(subconditions):
            return OrCondition(
                [
                    _ground_condition(subcondition, grounding, problem)
                    for subcondition in subconditions
                ]
            )
        case NotCondition(base_condition):
            return NotCondition(_ground_condition(base_condition, grounding, problem))
        case EqualityCondition(left_side, right_side):
            return EqualityCondition(
                _ground_argument(left_side, grounding),
                _ground_argument(right_side, grounding),
            )
        case ForallCondition(variables, condition):
            names = []      # list[variable_name]
            choices = []    # list[list[variable_grounding]]
            objects_by_type = {}
            for param in problem.objects_section:
                if param.type not in objects_by_type:
                    objects_by_type[param.type] = []
                objects_by_type[param.type].append(param.value)
            for typed_variable in variables:
                names.append(typed_variable.value)
                choices.append(objects_by_type.get(typed_variable.type, []))

            all_choices = itertools.product(*choices)
            grounded_conditions = []
            for choice in all_choices:
                # Copy grounding dict, and add in the specific choice
                # NOTE: this can explode memory if you make big foralls,
                # maybe this pipeline should be entirely generator based?
                # Would require rewriting the grounding and checking pipeline though.
                instance_grounding = dict(grounding)
                instance_grounding.update({name: value for name, value in zip(names, choice)})
                grounded_conditions.append(_ground_condition(condition, instance_grounding, problem))
            return AndCondition(grounded_conditions)
        case Predicate():
            return _ground_predicate(condition, grounding)


def _ground_effect(
    effect: Effect[Argument], grounding: Mapping[Variable, Object]
) -> Effect[Object]:
    match effect:
        case AndEffect(subeffects):
            return AndEffect(
                [
                    _ground_effect(subeffect, grounding)
                    for subeffect in subeffects
                ]
            )
        case ProbabilisticEffect():
            return ProbabilisticEffect(
                [
                    _ground_effect(possible_effect, grounding)
                    for possible_effect in effect._possible_effects
                ],
                effect._cummulative_probabilities,
            )
        case Predicate():
            return _ground_predicate(effect, grounding)
        case NotPredicate(base_predicate):
            return NotPredicate(
                cast(
                    Predicate[Object],
                    _ground_effect(base_predicate, grounding),
                )
            )


type Seed = int | float | str | bytes | bytearray | None
"""A seed for a simulation's RNG, powering its probabilistic aspects."""


@dataclass
class Simulation:
    """Low-level interface for PDDL simulation, backed by `SimulationState`.

    > [!NOTE]
    > For simply running simulations, locally, or with a server, prefer using
    > `pddlsim.remote.server`, or `pddlsim.local` for local simulations.

    `Simulation` provides additional functionality over `SimulationState`, such
    as getting all possible actions (`Simulation.get_grounded_actions`),
    checking if the problem has been solved (`Simulation.is_solved`), etc.

    `Simulation` is the low-level basis for other simulation interfaces
    in PDDLSIM. When interfacing with a full agent is undesired, it be a more
    lightweight alternative.

    The main way to construct a `Simulation` is via
    `Simulation.from_domain_and_problem`.
    """

    domain: Domain
    """The domain used in the simulation.
    
    Used in methods such as `Simulation.get_grounded_actions`.
    """
    problem: Problem
    """The problem used in the simulation.

    Used in methods such as `Simulation.is_solved`.
    """
    state: SimulationState
    """The current state of the simulation."""

    _rng: Random

    _reached_goal_indices: set[int]
    _unreached_goal_indices: set[int]
    _unactivated_revealables: set[Revealable]

    @cached_property
    def _object_name_id_allocator(self) -> IDAllocator[Object]:
        return IDAllocator.from_id_constructor(ObjectNameID)

    @cached_property
    def _predicate_id_allocator(self) -> IDAllocator[Identifier]:
        return IDAllocator.from_id_constructor(PredicateID)

    @cached_property
    def _type_name_id_allocator(self) -> IDAllocator[Type]:
        return IDAllocator.from_id_constructor(TypeNameID)

    @cached_property
    def _action_fallibilities(
        self,
    ) -> Mapping[Identifier, Iterable[ActionFallibility]]:
        fallibilities = defaultdict(list)

        for fallibility in self.problem.action_fallibilities_section:
            fallibilities[fallibility.grounded_action_schematic.name].append(
                fallibility
            )

        return fallibilities

    @cached_property
    def _objects_asp_part(self) -> ASPPart:
        return objects_asp_part(
            self.domain,
            self.problem,
            self._object_name_id_allocator,
            self._type_name_id_allocator,
        )

    @cached_property
    def _action_definition_asp_parts(
        self,
    ) -> Mapping[Identifier, tuple[ASPPart, IDAllocator[Variable]]]:
        return {
            action_definition.name: (
                action_definition_asp_part(
                    self.problem,
                    action_definition,
                    variable_id_allocator := IDAllocator[
                        Variable
                    ].from_id_constructor(VariableID),
                    self._object_name_id_allocator,
                    self._predicate_id_allocator,
                    self._type_name_id_allocator,
                ),
                variable_id_allocator,
            )
            for action_definition in self.domain.actions_section
        }

    @cached_property
    def _state_asp_part(self) -> ASPPart:
        return simulation_state_asp_part(
            self.state,
            self._predicate_id_allocator,
            self._object_name_id_allocator,
        )

    @classmethod
    def from_domain_and_problem(
        cls,
        domain: Domain,
        problem: Problem,
        state_override: SimulationState | None = None,
        reached_goal_indices_override: Iterable[int] | None = None,
        seed: Seed = None,
    ) -> "Simulation":
        """Construct a new `Simulation` from a domain and a problem.

        Additionally, the initial state for the simulation, as well as
        the already reached goals can be overrided. Finally, a seed for
        randomness in the simulation may be provided. When applying
        actions with probabilistic effects the seed is used for choosing
        a subeffect.
        """
        reached_goal_indices = (
            set(reached_goal_indices_override)
            if reached_goal_indices_override
            else set()
        )

        return Simulation(
            domain,
            problem,
            # Internally, we mutate the state, so copying is needed
            state_override._copy()
            if state_override
            else SimulationState(
                {
                    true_predicate
                    for true_predicate in problem.initialization_section
                }
            ),
            # Technically speaking, the seed could be cracked under very
            # specific circumstances (system time is known and is used
            # for randomness), but in practice, this is fine, and shouldn't
            # be exploited.
            Random(seed),
            reached_goal_indices,
            set(range(len(problem.goals_section))) - reached_goal_indices,
            set(problem.revealables_section),
        )

    def __post_init__(self) -> None:
        """Run update procedures that normally run on a successful action.

        These should also be run on the first state of the simulation
        (revealables, reached goals, etc.)
        """
        self._update_reached_goals()
        self._update_revealables()

    def _update_reached_goals(self) -> None:
        newly_reached_goals = set()

        for goal_index in self._unreached_goal_indices:
            if self.state.does_condition_hold(
                self.problem.goals_section[goal_index]
            ):
                self._reached_goal_indices.add(goal_index)
                newly_reached_goals.add(goal_index)

        self._unreached_goal_indices.difference_update(newly_reached_goals)

    def _update_revealables(self) -> None:
        newly_active_revealables = set()

        while True:
            for revealable in self._unactivated_revealables:
                if self.state.does_condition_hold(revealable.condition):
                    should_reveal = (
                        self._rng.random() < revealable.with_probability
                    )

                    if should_reveal:
                        self.state._make_effect_hold(
                            revealable.effect, self._rng
                        )
                        newly_active_revealables.add(revealable)

            if newly_active_revealables:
                self._unactivated_revealables.difference_update(
                    newly_active_revealables
                )

                newly_active_revealables.clear()
            else:
                break

    @property
    def reached_goal_indices(self) -> list[int]:
        """Indices of completed problem goals, from 0, in definition order."""
        return list(self._reached_goal_indices)

    @property
    def unreached_goal_indices(self) -> list[int]:
        """Indices of uncompleted problem goals, from 0, in definition order."""
        return list(self._unreached_goal_indices)

    def apply_grounded_action(self, grounded_action: GroundedAction) -> bool:
        """Apply a grounded action to the `Simulation`, affecting its state.

        The returned boolean represents if the action was successful, or failed.
        Grounded actions that are invalid will raise a `ValueError`, and not
        just fail.

        If the problem the simulation has any action fallibilities, these may
        apply, making the action have no effect. If the action has as a
        subeffect a probabilistic one, a subeffect is chosen at random based on
        the simulation's random number generator.
        """
        for fallibility in self._action_fallibilities[grounded_action.name]:
            if fallibility.grounded_action_schematic.does_match(
                grounded_action
            ) and self.state.does_condition_hold(fallibility.condition):
                does_fail = self._rng.random() < fallibility.with_probability

                if does_fail:
                    return False

        action_definition = self.domain.actions_section[grounded_action.name]
        grounding = {
            variable: object_
            for variable, object_ in zip(
                (parameter.value for parameter in action_definition.parameters),
                grounded_action.grounding,
                strict=True,
            )
        }

        if not self.state.does_condition_hold(
            _ground_condition(action_definition.precondition, grounding, self.problem)
        ):
            raise ValueError("grounded action doesn't satisfy precondition")

        self.state._make_effect_hold(
            _ground_effect(action_definition.effect, grounding),
            self._rng,
        )

        if hasattr(self, "_state_asp_part"):
            del self._state_asp_part  # Regenerate the cached state ASP part

        self._update_reached_goals()
        self._update_revealables()

        return True


    def _get_groundings(
        self, action_definition: ActionDefinition
    ) -> Generator[Mapping[Variable, Object]]:
        action_definition_asp_part, variable_id_allocator = (
            self._action_definition_asp_parts[action_definition.name]
        )

        # `-Wno-atom-undefined` disables warnings about undefined atoms
        # from Clingo. This is useful, as for some simulation states,
        # a predicate has no valid assignments, and won't show up
        # in the ASP program.
        control = Control(["-Wno-atom-undefined"])

        # Set number of threads to use
        control.configuration.solve.parallel_mode = os.cpu_count()  # type: ignore
        # Compute all models (all groundings)
        control.configuration.solve.models = 0  # type: ignore

        self._objects_asp_part.add_to_control(control)
        self._state_asp_part.add_to_control(control)
        action_definition_asp_part.add_to_control(control)

        control.ground(
            (
                (ASPPartKind.OBJECTS, ()),
                (ASPPartKind.STATE, ()),
                (ASPPartKind.ACTION_DEFINITION, ()),
            )
        )

        with control.solve(yield_=True) as handle:
            for model in handle:
                yield {
                    variable_id_allocator.get_value(
                        VariableID.from_str(symbol.name)
                    ): self._object_name_id_allocator.get_value(
                        ObjectNameID.from_str(symbol.arguments[0].name)
                    )
                    for symbol in model.symbols(shown=True)
                }

    def _get_grounded_actions(
        self, action_definition: ActionDefinition
    ) -> Iterable[GroundedAction]:
        return (
            GroundedAction(
                action_definition.name,
                tuple(
                    grounding[parameter.value]
                    for parameter in action_definition.parameters
                ),
            )
            for grounding in self._get_groundings(action_definition)
        )

    def get_grounded_actions(self) -> Iterable[GroundedAction]:
        """Get possible grounded actions for the current simulation state."""
        return (
            grounded_action
            for action_definition in self.domain.actions_section
            for grounded_action in self._get_grounded_actions(action_definition)
        )

    def is_solved(self) -> bool:
        """Check if all goals of the problem have been achieved."""
        return len(self._reached_goal_indices) == len(
            self.problem.goals_section
        )
