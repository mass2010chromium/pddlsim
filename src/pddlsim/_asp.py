from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, MutableMapping, Sequence, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain
from typing import NewType

import clingo.ast
from clingo import Control

from pddlsim.ast import (
    ActionDefinition,
    AndCondition,
    Argument,
    Condition,
    Domain,
    EqualityCondition,
    ForallCondition,
    Identifier,
    NotCondition,
    Object,
    OrCondition,
    Predicate,
    Problem,
    Type,
    Variable,
)
from pddlsim.state import SimulationState


@dataclass(frozen=True, eq=True)
class ID(ABC):
    value: int

    @classmethod
    @abstractmethod
    def prefix(cls) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.prefix()}{self.value}"

    @classmethod
    def from_str(cls, string: str) -> "ID":
        if string.startswith(cls.prefix()):
            return cls(int(string[len(cls.prefix()) :]))

        raise ValueError("id prefix not recognized")


@dataclass(frozen=True, eq=True)
class VariableID(ID):
    @classmethod
    def prefix(cls) -> str:
        return "variable"


@dataclass(frozen=True, eq=True)
class ObjectNameID(ID):
    @classmethod
    def prefix(cls) -> str:
        return "object"


@dataclass(frozen=True, eq=True)
class PredicateID(ID):
    @classmethod
    def prefix(cls) -> str:
        return "predicate"


@dataclass(frozen=True, eq=True)
class TypeNameID(ID):
    @classmethod
    def prefix(cls) -> str:
        return "type"


@dataclass(frozen=True, eq=True)
class RuleID(ID):
    @classmethod
    def prefix(cls) -> str:
        return "rule"


@dataclass(frozen=True, eq=True)
class TemporaryID(ID):
    @classmethod
    def prefix(cls) -> str:
        return "T"


@dataclass
class IDAllocator[T]:
    _previous_id: int
    _new_id: Callable[[int], ID]
    _ids: MutableMapping[T, ID]
    _values: MutableMapping[ID, T]

    @classmethod
    def from_id_constructor(
        cls, new_id: Callable[[int], ID]
    ) -> "IDAllocator[T]":
        return IDAllocator(-1, new_id, {}, {})

    def next_id(self) -> ID:
        self._previous_id += 1

        return self._new_id(self._previous_id)

    def __iter__(self) -> Generator[tuple[T, ID]]:
        yield from self._ids.items()

    def get_id_or_insert(self, value: T) -> ID:
        if value in self._ids:
            return self._ids[value]
        else:
            id = self.next_id()

            self._ids[value] = id
            self._values[id] = value

            return id

    def get_value(self, id: ID) -> T:
        return self._values[id]


# Used to enforce parameters to methods on `ASPPart` are correct
SymbolAST = NewType("SymbolAST", clingo.ast.AST)
VariableAST = NewType("VariableAST", clingo.ast.AST)
type ArgumentAST = SymbolAST | VariableAST
LiteralAST = NewType("LiteralAST", clingo.ast.AST)


@dataclass(frozen=True)
class ASPPart:
    name: str
    _statements: list[clingo.ast.AST] = field(default_factory=list)

    def next_location(self) -> clingo.ast.Location:
        # We add one to the length as lines start at 1
        position = clingo.ast.Position("<ast>", len(self._statements) + 1, 1)

        return clingo.ast.Location(position, position)

    def __post_init__(self) -> None:
        self._statements.append(
            clingo.ast.Program(self.next_location(), self.name, [])
        )

    def create_symbol(self, name: str) -> SymbolAST:
        return SymbolAST(
            clingo.ast.SymbolicTerm(
                self.next_location(), clingo.Function(name, [])
            )
        )

    def create_variable(self, name: str) -> VariableAST:
        return VariableAST(clingo.ast.Variable(self.next_location(), name))

    def _create_literal(
        self, atom: clingo.ast.AST, truthiness: bool = True
    ) -> LiteralAST:
        return LiteralAST(
            clingo.ast.Literal(
                self.next_location(),
                clingo.ast.Sign.NoSign
                if truthiness
                else clingo.ast.Sign.Negation,
                atom,
            )
        )

    def create_function_literal(
        self,
        name: str,
        arguments: Sequence[ArgumentAST],
        truthiness: bool = True,
    ) -> LiteralAST:
        return self._create_literal(
            clingo.ast.SymbolicAtom(
                clingo.ast.Function(
                    self.next_location(), name, arguments, False
                ),
            ),
            truthiness,
        )

    def create_equality_literal(
        self, left_side: ArgumentAST, right_side: ArgumentAST
    ) -> LiteralAST:
        return self._create_literal(
            clingo.ast.Comparison(
                left_side,
                [
                    clingo.ast.Guard(
                        clingo.ast.ComparisonOperator.Equal, right_side
                    )
                ],
            ),
            True,
        )

    def create_constant_literal(
        self, name: str, truthiness: bool = True
    ) -> LiteralAST:
        return self.create_function_literal(name, [], truthiness)

    def _add_fact(self, ast: clingo.ast.AST) -> None:
        self._statements.append(clingo.ast.Rule(self.next_location(), ast, []))

    def add_fact(self, ast: LiteralAST) -> None:
        self._add_fact(ast)

    def add_rule(self, head: LiteralAST, body: Sequence[LiteralAST]) -> None:
        self._statements.append(
            clingo.ast.Rule(self.next_location(), head, body)
        )

    def add_integrity_constraint(self, body: Sequence[LiteralAST]) -> None:
        self.add_rule(
            self._create_literal(clingo.ast.BooleanConstant(False), True),
            body,
        )

    def add_single_instantiation_constraint(
        self, literal: LiteralAST, conditions: Sequence[LiteralAST]
    ) -> None:
        self._add_fact(
            clingo.ast.Aggregate(
                self.next_location(),
                clingo.ast.Guard(
                    clingo.ast.ComparisonOperator.Equal,
                    clingo.ast.SymbolicTerm(
                        self.next_location(), clingo.Number(1)
                    ),
                ),
                [
                    clingo.ast.ConditionalLiteral(
                        self.next_location(), literal, conditions
                    )
                ],
                clingo.ast.Guard(
                    clingo.ast.ComparisonOperator.Equal,
                    clingo.ast.SymbolicTerm(
                        self.next_location(), clingo.Number(1)
                    ),
                ),
            )
        )

    def add_show_signature(self, name: str, arity: int) -> None:
        self._statements.append(
            clingo.ast.ShowSignature(self.next_location(), name, arity, True)
        )

    def add_to_control(self, control: Control) -> None:
        with clingo.ast.ProgramBuilder(control) as builder:
            for statement in self._statements:
                builder.add(statement)


class ASPPartKind(StrEnum):
    OBJECTS = "objects"
    STATE = "state"
    ACTION_DEFINITION = "action_definition"


def objects_asp_part(
    domain: Domain,
    problem: Problem,
    object_id_allocator: IDAllocator[Object],
    type_id_allocator: IDAllocator[Type],
) -> ASPPart:
    part = ASPPart(ASPPartKind.OBJECTS)

    for object_ in chain(problem.objects_section, domain.constants_section):
        type_id = type_id_allocator.get_id_or_insert(object_.type)
        object_id = object_id_allocator.get_id_or_insert(object_.value)

        part.add_fact(
            part.create_function_literal(
                str(type_id), [part.create_symbol(str(object_id))]
            )
        )

    for member in domain.types_section:
        custom_type = member.value
        supertype = member.type

        custom_type_id = type_id_allocator.get_id_or_insert(custom_type)
        supertype_id = type_id_allocator.get_id_or_insert(supertype)

        part.add_rule(
            part.create_function_literal(
                str(supertype_id), [part.create_variable("O")]
            ),
            [
                part.create_function_literal(
                    str(custom_type_id), [part.create_variable("O")]
                )
            ],
        )

    return part


def simulation_state_asp_part(
    state: SimulationState,
    predicate_id_allocator: IDAllocator[Identifier],
    object_id_allocator: IDAllocator[Object],
) -> ASPPart:
    part = ASPPart(ASPPartKind.STATE)

    for predicate in state._true_predicates:
        predicate_id = predicate_id_allocator.get_id_or_insert(predicate.name)

        part.add_fact(
            part.create_function_literal(
                str(predicate_id),
                [
                    part.create_symbol(
                        str(object_id_allocator.get_id_or_insert(object_))
                    )
                    for object_ in predicate.assignment
                ],
            )
        )

    return part


def _add_condition_to_asp_part(
    condition: Condition[Argument],
    part: ASPPart,
    rule_id_allocator: IDAllocator,
    variable_id_allocator: IDAllocator[Variable],
    object_id_allocator: IDAllocator[Object],
    predicate_id_allocator: IDAllocator[Identifier],
) -> ID:
    temporary_id_allocator = IDAllocator[Variable].from_id_constructor(
        TemporaryID
    )

    def argument_to_asp(argument: Argument) -> ArgumentAST:
        match argument:
            case Variable():
                temporary_id = temporary_id_allocator.get_id_or_insert(argument)
                return part.create_variable(str(temporary_id))
            case Object():
                return part.create_symbol(
                    str(object_id_allocator.get_id_or_insert(argument))
                )

    rule_id = rule_id_allocator.next_id()
    head = part.create_constant_literal(str(rule_id))

    match condition:
        case AndCondition(subconditions):
            subcondition_ids = (
                _add_condition_to_asp_part(
                    subcondition,
                    part,
                    rule_id_allocator,
                    variable_id_allocator,
                    object_id_allocator,
                    predicate_id_allocator,
                )
                for subcondition in subconditions
            )

            part.add_rule(
                head,
                [
                    part.create_constant_literal(str(subcondition_id))
                    for subcondition_id in subcondition_ids
                ],
            )
        case OrCondition(subconditions):
            for subcondition in subconditions:
                subcondition_id = _add_condition_to_asp_part(
                    subcondition,
                    part,
                    rule_id_allocator,
                    variable_id_allocator,
                    object_id_allocator,
                    predicate_id_allocator,
                )

                part.add_rule(
                    head,
                    [part.create_constant_literal(str(subcondition_id))],
                )
        case NotCondition(base_condition):
            base_condition_id = _add_condition_to_asp_part(
                base_condition,
                part,
                rule_id_allocator,
                variable_id_allocator,
                object_id_allocator,
                predicate_id_allocator,
            )

            part.add_rule(
                head,
                [part.create_constant_literal(str(base_condition_id), False)],
            )
        case Predicate(name=name, assignment=assignment):
            predicate_id = predicate_id_allocator.get_id_or_insert(name)

            predicate_literal = part.create_function_literal(
                str(predicate_id),
                [argument_to_asp(argument) for argument in assignment],
            )

            body = [
                part.create_function_literal(
                    str(variable_id_allocator.get_id_or_insert(variable)),
                    [part.create_variable(str(temporary_id))],
                )
                for variable, temporary_id in temporary_id_allocator
            ]

            body.append(predicate_literal)

            part.add_rule(
                head,
                body,
            )
        case EqualityCondition(left_side=left_side, right_side=right_side):
            left_side_ast = argument_to_asp(left_side)
            right_side_ast = argument_to_asp(right_side)

            equality_literal = part.create_equality_literal(
                left_side_ast, right_side_ast
            )

            body = [
                part.create_function_literal(
                    str(variable_id_allocator.get_id_or_insert(variable)),
                    [part.create_variable(str(temporary_id))],
                )
                for variable, temporary_id in temporary_id_allocator
            ]

            body.append(equality_literal)

            part.add_rule(
                head,
                body,
            )

    return rule_id


def action_definition_asp_part(
    problem: Problem,
    action_definition: ActionDefinition,
    variable_id_allocator: IDAllocator[Variable],
    object_id_allocator: IDAllocator[Object],
    predicate_id_allocator: IDAllocator[Identifier],
    type_id_allocator: IDAllocator[Type],
) -> ASPPart:
    part = ASPPart(ASPPartKind.ACTION_DEFINITION)

    # Require each parameter have a single object as its value
    for parameter in action_definition.parameters:
        variable_id = variable_id_allocator.get_id_or_insert(parameter.value)
        type_id = type_id_allocator.get_id_or_insert(parameter.type)

        part.add_single_instantiation_constraint(
            part.create_function_literal(
                str(variable_id), [part.create_variable("O")]
            ),
            [
                part.create_function_literal(
                    str(type_id), [part.create_variable("O")]
                )
            ],
        )

    # Preprocess precondition to replace Foralls with series of Ands
    def _expand_substitute(
        condition: Condition[Argument], substitutions: Mapping[Variable, Object]
    ) -> Condition[Object]:
        """Fold in to use global information for Forall..."""
        match condition:
            case AndCondition(subconditions):
                return AndCondition(
                    [
                        _expand_substitute(subcondition, substitutions)
                        for subcondition in subconditions
                    ]
                )
            case OrCondition(subconditions):
                return OrCondition(
                    [
                        _expand_substitute(subcondition, substitutions)
                        for subcondition in subconditions
                    ]
                )
            case NotCondition(base_condition):
                return NotCondition(_expand_substitute(base_condition, substitutions))
            case EqualityCondition(left_side, right_side):
                return EqualityCondition(
                    substitutions.get(left_side, left_side),
                    substitutions.get(right_side, right_side),
                )
            case ForallCondition(loop_var, condition):
                new_subs = dict(substitutions)
                conditions = []
                for param in problem.objects_section:
                    if param.type == loop_var.type:
                        new_subs[loop_var.value] = param.value
                        conditions.append(_expand_substitute(condition, new_subs))
                return AndCondition(conditions)
            case Predicate(name, assignment):
                return Predicate(
                    name,
                    tuple(
                        substitutions.get(argument, argument)
                        for argument in assignment
                    ),
                )

    precondition_id = _add_condition_to_asp_part(
        _expand_substitute(action_definition.precondition, {}),
        part,
        IDAllocator.from_id_constructor(RuleID),
        variable_id_allocator,
        object_id_allocator,
        predicate_id_allocator,
    )
    part.add_integrity_constraint(
        [part.create_constant_literal(str(precondition_id), False)]
    )

    # Show in the model only the variables
    for _, variable_id in variable_id_allocator:
        part.add_show_signature(str(variable_id), 1)

    return part
