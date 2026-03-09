"""Contains utilities to construct `Domain` and `Problem` objects from text."""

import os
from collections.abc import Iterable
from decimal import Decimal
from itertools import chain
from typing import cast

from lark import Lark, Token, Transformer, v_args

from pddlsim import _RESOURCES
from pddlsim.ast import (
    ActionDefinition,
    ActionFallibilitiesSection,
    ActionFallibility,
    ActionsSection,
    AndCondition,
    AndEffect,
    Argument,
    Condition,
    ConstantsSection,
    CustomType,
    Domain,
    Effect,
    EqualityCondition,
    FileLocation,
    ForallCondition,
    GoalsSection,
    GroundedActionSchematic,
    Identifier,
    InitializationSection,
    Location,
    NotCondition,
    NotPredicate,
    Object,
    ObjectsSection,
    ObjectType,
    OrCondition,
    Parameters,
    Predicate,
    PredicateDefinition,
    PredicatesSection,
    ProbabilisticEffect,
    Problem,
    RawProblem,
    Requirement,
    RequirementsSection,
    Revealable,
    RevealablesSection,
    Type,
    Typed,
    TypesSection,
    Variable,
)


@v_args(inline=True)
class _PDDLTransformer(Transformer):
    def NUMBER(self, token: Token) -> Decimal:  # noqa: N802
        return Decimal(token)

    def IDENTIFIER(self, token: Token) -> Identifier:  # noqa: N802
        return Identifier(str(token), location=FileLocation._from_token(token))

    def VARIABLE(self, token: Token) -> Variable:  # noqa: N802
        return Variable(token[1:], location=FileLocation._from_token(token))

    @v_args(inline=False)
    def list_[T](self, items: list[T]) -> list[T]:
        return items

    @v_args(inline=False)
    def nonempty_list[T](self, items: list[T]) -> list[T]:
        return items

    def strips_requirement(self) -> Requirement:
        return Requirement.STRIPS

    def typing_requirement(self) -> Requirement:
        return Requirement.TYPING

    def disjunctive_preconditions_requirement(self) -> Requirement:
        return Requirement.DISJUNCTIVE_PRECONDITIONS

    def negative_preconditions_requirement(self) -> Requirement:
        return Requirement.NEGATIVE_PRECONDITIONS

    def equality_requirement(self) -> Requirement:
        return Requirement.EQUALITY

    def probabilistic_effects(self) -> Requirement:
        return Requirement.PROBABILISTIC_EFFECTS

    def fallible_actions_requirement(self) -> Requirement:
        return Requirement.FALLIBLE_ACTIONS

    def revealables_requirement(self) -> Requirement:
        return Requirement.REVEALABLES

    def multiple_goals_requirement(self) -> Requirement:
        return Requirement.MULTIPLE_GOALS

    def REQUIREMENTS_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def requirements_section(
        self, location: Location, requirements: list[Requirement]
    ) -> RequirementsSection:
        return RequirementsSection.from_raw_parts(
            requirements, location=location
        )

    def object_type(self) -> ObjectType:
        return ObjectType()

    def custom_type(self, identifier: Identifier) -> CustomType:
        return CustomType(identifier.value, location=identifier.location)

    def typed_list_part[T](
        self, items: list[T], type: Type
    ) -> Iterable[Typed[T]]:
        return (Typed(item, type) for item in items)

    def object_typed_list[T](self, items: Iterable[T]) -> Iterable[Typed[T]]:
        return (Typed(item, ObjectType()) for item in items)

    def typed_list[T](
        self,
        head: Iterable[Typed[T]],
        tail: Iterable[Typed[T]] | None,
    ) -> Iterable[Typed[T]]:
        return chain(head, tail) if tail else head

    def TYPES_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def types_section(
        self, location: Location, types: Iterable[Typed[CustomType]]
    ) -> TypesSection:
        return TypesSection.from_raw_parts(types, location=location)

    def object_(self, identifier: Identifier) -> Object:
        return Object(identifier.value, location=identifier.location)

    def constants_section(
        self, objects: list[Typed[Object]]
    ) -> ConstantsSection:
        return ConstantsSection.from_raw_parts(objects)

    def predicate_definition(
        self,
        name: Identifier,
        parameters: Iterable[Typed[Variable]],
    ) -> PredicateDefinition:
        return PredicateDefinition.from_raw_parts(
            name, Parameters.from_raw_parts(parameters, definition=name)
        )

    def predicates_section(
        self,
        predicate_definitions: list[PredicateDefinition],
    ) -> PredicatesSection:
        return PredicatesSection.from_raw_parts(predicate_definitions)

    def predicate[A: Argument](
        self,
        name: Identifier,
        assignment: list[A],
    ) -> Predicate[A]:
        return Predicate(name, tuple(assignment))

    def and_condition[A: Argument](
        self, operands: list[Condition[A]]
    ) -> AndCondition[A]:
        return AndCondition(operands)

    def OR_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def or_condition[A: Argument](
        self, location: Location, operands: list[Condition[A]]
    ) -> OrCondition[A]:
        return OrCondition(operands, location=location)

    def NOT_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def not_condition[A: Argument](
        self, location: Location, operand: Condition[A]
    ) -> NotCondition[A]:
        return NotCondition(operand, location=location)

    def EQUALS_SIGN(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def equality_condition[A: Argument](
        self,
        location: Location,
        left_side: A,
        right_side: A,
    ) -> EqualityCondition[A]:
        return EqualityCondition(left_side, right_side, location=location)

    def forall_condition[A: Argument](
        self,
        parameters: Iterable[Typed[Variable]],
        condition: Condition[A],
    ) -> ForallCondition[A]:
        return ForallCondition(list(parameters), condition)

    def not_predicate[A: Argument](
        self, base_predicate: Predicate[A]
    ) -> NotPredicate[A]:
        return NotPredicate(base_predicate)

    def and_effect[A: Argument](
        self, subeffects: list[Effect[A]]
    ) -> AndEffect[A]:
        return AndEffect(subeffects)

    def probabilistic_effect_pair[A: Argument](
        self, probability: Decimal, effect: Effect[A]
    ) -> tuple[Decimal, Effect[A]]:
        return (probability, effect)

    def PROBABILISTIC_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def probabilistic_effect[A: Argument](
        self,
        location: Location,
        possibilities: list[tuple[Decimal, Effect[A]]],
    ) -> ProbabilisticEffect:
        return ProbabilisticEffect.from_possibilities(
            possibilities, location=location
        )

    def action_definition(
        self,
        name: Identifier,
        parameters: list[Typed[Variable]],
        precondition: Condition[Argument] | None,
        effect: Effect[Argument] | None,
    ) -> ActionDefinition:
        return ActionDefinition.from_raw_parts(
            name,
            Parameters.from_raw_parts(parameters, definition=name),
            precondition if precondition else AndCondition([]),
            effect if effect else AndEffect([]),
        )

    def actions_section(
        self, action_definitions: list[ActionDefinition]
    ) -> ActionsSection:
        return ActionsSection.from_raw_parts(action_definitions)

    def domain(
        self,
        name: Identifier,
        requirements: RequirementsSection | None,
        type_hierarchy: TypesSection | None,
        constants: ConstantsSection | None,
        predicate_definitions: PredicatesSection | None,
        action_definitions: ActionsSection | None,
    ) -> Domain:
        return Domain.from_raw_parts(
            name,
            requirements if requirements else RequirementsSection({}),  # type: ignore
            type_hierarchy,
            constants if constants else ConstantsSection(),
            predicate_definitions
            if predicate_definitions
            else PredicatesSection(),
            action_definitions if action_definitions else ActionsSection(),
        )

    def objects_section(self, objects: list[Typed[Object]]) -> ObjectsSection:
        return ObjectsSection.from_raw_parts(objects)

    def grounded_action_schematic(
        self, name: Identifier, grounding: list[Argument]
    ) -> GroundedActionSchematic:
        return GroundedActionSchematic(name, tuple(grounding))

    def ACTION_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def action_fallibility(
        self,
        location: Location,
        grounded_action_schematic: GroundedActionSchematic,
        with_probability: Decimal,
        condition: Condition[Object],
    ) -> ActionFallibility:
        return ActionFallibility(
            grounded_action_schematic,
            condition,
            with_probability,
            location=location,
        )

    def FAIL_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def action_fallibilities_section(
        self, location: Location, fallibilities: list[ActionFallibility]
    ) -> ActionFallibilitiesSection:
        return ActionFallibilitiesSection(fallibilities, location=location)

    def WHEN_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def revealable(
        self,
        location: Location,
        with_probability: Decimal | None,
        condition: Condition[Object],
        effect: Effect[Object],
    ) -> Revealable:
        return Revealable(
            effect,
            condition,
            with_probability if with_probability else Decimal(value=1),
            location=location,
        )

    def REVEAL_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def revealables_section(
        self, location: Location, revealables: list[Revealable]
    ) -> RevealablesSection:
        return RevealablesSection(revealables, location=location)

    def initialization_section(
        self, predicates: list[Predicate[Object]]
    ) -> InitializationSection:
        return InitializationSection.from_raw_parts(predicates)

    def GOALS_KEYWORD(self, token: Token) -> Location:  # noqa: N802
        return FileLocation._from_token(token)

    def goals_section(
        self, location: Location, goals: list[Condition[Object]]
    ) -> GoalsSection:
        return GoalsSection(goals, location=location)

    def problem(
        self,
        name: Identifier,
        used_domain_name: Identifier,
        requirements_section: RequirementsSection | None,
        objects_section: ObjectsSection | None,
        action_fallibilities_section: ActionFallibilitiesSection | None,
        revealables_section: RevealablesSection | None,
        initialization_section: InitializationSection | None,
        goals_section: GoalsSection | Condition[Object],
    ) -> RawProblem:
        return RawProblem.from_raw_parts(
            name,
            used_domain_name,
            requirements_section
            if requirements_section
            else RequirementsSection(),
            objects_section if objects_section else ObjectsSection(),
            action_fallibilities_section,
            revealables_section,
            initialization_section
            if initialization_section
            else InitializationSection(),
            goals_section,
        )


# Cache the parser for each invocation, and persist it
_PDDL_PARSER = Lark(
    _RESOURCES.joinpath("grammar.lark").read_text(),
    parser="lalr",
    cache=True,
    transformer=_PDDLTransformer(),
    start=["domain", "problem"],
)


def parse_domain(text: str) -> Domain:
    """Construct a `pddlsim.ast.Domain` from PDDL text."""
    return cast(Domain, _PDDL_PARSER.parse(text, "domain"))


def parse_problem(text: str, domain: Domain) -> Problem:
    """Construct a `pddlsim.ast.Problem` from PDDL text.

    Due to validation concerns, this function requires an existing
    `pddlsim.ast.Domain` object, corresponding to the domain used
    in the problem.
    """
    return Problem(
        cast(
            RawProblem,
            _PDDL_PARSER.parse(text, "problem"),
        ),
        domain,
    )


def parse_domain_problem_pair(
    domain_text: str, problem_text: str
) -> tuple[Domain, Problem]:
    """Construct a `pddlsim.ast.Domain` and a `pddlsim.ast.Problem` from text.

    This is a convenience function to avoid manually passing a
    `pddlsim.ast.Domain` into `parse_problem`.
    """
    domain = parse_domain(domain_text)
    problem = parse_problem(problem_text, domain)

    return (domain, problem)


def parse_domain_from_file(path: str | os.PathLike) -> Domain:
    """Construct a `pddlsim.ast.Domain` from the path to a file.

    This is a convenience function to avoid manual I/O.
    """
    with open(path) as file:
        return parse_domain(file.read())


def parse_problem_from_file(path: str | os.PathLike, domain: Domain) -> Problem:
    """Construct a `pddlsim.ast.Problem` from the path to a file.

    This is a convenience function to avoid manual I/O. Like `parse_problem`, it
    requires manually passing the `pddlsim.ast.Domain` object corresponding to
    the domain used in the problem, for validation of the problem.
    """
    with open(path) as file:
        return parse_problem(file.read(), domain)


def parse_domain_problem_pair_from_files(
    domain_path: str | os.PathLike, problem_path: str | os.PathLike
) -> tuple[Domain, Problem]:
    """Construct a `pddlsim.ast.Domain` and a `pddlsim.ast.Problem` from paths.

    This is a convenience function to avoid manual I/O. Like
    `parse_domain_problem_pair`, it mainly exists to avoid passing a
    `pddlsim.ast.Domain` manually into `parse_problem_from_file`.
    """
    domain = parse_domain_from_file(domain_path)
    problem = parse_problem_from_file(problem_path, domain)

    return (domain, problem)
