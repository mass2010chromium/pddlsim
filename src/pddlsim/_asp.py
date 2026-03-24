from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, MutableMapping, Sequence, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import itertools
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

ASP_TRUE = "dummy_true"
ASP_ACTION_LABEL = "action"

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

    def create_conditional_literal(
        self, literal: LiteralAST, conditions: Sequence[LiteralAST]
    ) -> LiteralAST:
        return clingo.ast.ConditionalLiteral(
            self.next_location(),
            literal,
            conditions
        )

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
        # A set of "dummy facts" to absorb unused arguments with.
        # Basially, a universal true statement that absorbs the argument.
        part.add_fact(
            part.create_function_literal(
                ASP_TRUE, [part.create_symbol(str(object_id))]
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
    type_id_allocator: IDAllocator[Type],
) -> tuple[ID, list[Variable]]:

    """
    Add rules corresponding to a precondition of an action in PDDL.

    Preconditions are added like so:

    (predicate X Y Z):
      -> rule1(X, Y, Z) :- predicate(X, Y, Z).

    (and (A) (B)):
      -> rule(X, Y) :- ruleA(X), ruleB(Y).

    (or (A) (B)):
      -> rule(X, Y) :- ruleA(X), true(Y).
         rule(X, Y) :- true(X), ruleB(Y).
    
    (not (A)):
      -> rule(X) :- true(X), not ruleA(X).

    (= X Y):
      -> rule(X, Y) :- true(X), true(Y), X = Y.

    (forall (X, Y) (C)):
      -> rule(A) :- true(Z), ruleC(X, Y, Z) : X, Y.

    where A, B, C are conditions, depending on variables X, Y, Z respectively,

    where the exact rule namings are slightly different in the code, but translated
      to ruleA/ruleB/ruleC to match the A, B, C conditions,

    where the true(*) facts are just always true for any object
      (defined in object asp section).

    Loosely. The actual arguments will depend on the number of used
    variables in the subconditions, and forall quantification variables
    are specified using their type facts.
    

    Returns a pair (added rule ID, used variables, in call order).
    """

    def argument_to_asp(argument: Argument) -> ArgumentAST:
        match argument:
            case Variable():   # value is the name
                return part.create_variable(
                    str(variable_id_allocator.get_id_or_insert(argument))
                )
            case Object():
                return part.create_symbol(
                    str(object_id_allocator.get_id_or_insert(argument))
                )

    def make_function_literal(
            identifier: ID | str,
            arguments: list[Argument],
            truthiness: bool = True
    ) -> LiteralAST:
        # Shorthand to avoid rewriting this listcomp over and over again
        return part.create_function_literal(
            str(identifier),
            [argument_to_asp(arg) for arg in arguments],
            truthiness
        )

    rule_id = rule_id_allocator.next_id()

    match condition:
        case AndCondition(subconditions):
            subcondition_ids, subcondition_used_variables = list(zip(*(
                # Each of these calls returns a pair. These are collected
                #   as the arguments to zip(), which puts all the IDs into
                #   one list, and the variable sets into the second.
                _add_condition_to_asp_part(
                    subcondition,
                    part,
                    rule_id_allocator,
                    variable_id_allocator,
                    object_id_allocator,
                    predicate_id_allocator,
                    type_id_allocator
                )
                for subcondition in subconditions
            )))

            used_variables: list[Variable] = list(set().union(
                *(set(variables) for variables in subcondition_used_variables)
            ))
            head = make_function_literal(rule_id, used_variables)

            part.add_rule(
                head,
                [
                    make_function_literal(subcondition_id, sub_used_variables)
                    for subcondition_id, sub_used_variables in zip(
                        subcondition_ids, subcondition_used_variables
                    )
                ],
            )
        case OrCondition(subconditions):
            subcondition_ids, subcondition_used_variables = list(zip(*(
                # Each of these calls returns a pair. These are collected
                #   as the arguments to zip(), which puts all the IDs into
                #   one list, and the variable sets into the second.
                _add_condition_to_asp_part(
                    subcondition,
                    part,
                    rule_id_allocator,
                    variable_id_allocator,
                    object_id_allocator,
                    predicate_id_allocator,
                    type_id_allocator
                )
                for subcondition in subconditions
            )))

            used_variables: list[Variable] = list(set().union(
                *(set(variables) for variables in subcondition_used_variables)
            ))
            head = make_function_literal(rule_id, used_variables)

            set_used_variables = set(used_variables)

            for subcondition_id, sub_used_variables in zip(
                subcondition_ids, subcondition_used_variables
            ):
                need_absorbing = set_used_variables - set(sub_used_variables)
                part.add_rule(
                    head,
                    [
                        *(
                            make_function_literal(ASP_TRUE, [variable])
                            for variable in need_absorbing
                        ),
                        make_function_literal(subcondition_id, sub_used_variables)
                    ]
                )

        case NotCondition(base_condition):
            base_condition_id, used_variables = _add_condition_to_asp_part(
                base_condition,
                part,
                rule_id_allocator,
                variable_id_allocator,
                object_id_allocator,
                predicate_id_allocator,
                type_id_allocator
            )
            head = make_function_literal(rule_id, used_variables)
            part.add_rule(
                head,
                [
                    *(
                        make_function_literal(ASP_TRUE, [variable])
                        for variable in used_variables 
                    ),
                    make_function_literal(base_condition_id, used_variables, False)
                ]
            )

        case Predicate(name=name, assignment=assignment):
            # NOTE: it's possible that new (never true) predicates need to be inserted
            # at condition parse time
            predicate_id = predicate_id_allocator.get_id_or_insert(name)

            predicate_literal = make_function_literal(predicate_id, assignment)
            used_variables = list(filter(lambda x: type(x) == Variable, assignment))

            head = make_function_literal(rule_id, used_variables)
            part.add_rule(
                head,
                [predicate_literal],
            )
        case EqualityCondition(left_side=left_side, right_side=right_side):
            left_side_ast = argument_to_asp(left_side)
            right_side_ast = argument_to_asp(right_side)

            equality_literal = part.create_equality_literal(
                left_side_ast, right_side_ast
            )
            used_variables = list(filter(lambda x: type(x) == Variable, [left_side, right_side]))

            head = make_function_literal(rule_id, used_variables)
            part.add_rule(
                head,
                [
                    *(
                        make_function_literal(ASP_TRUE, [variable])
                        for variable in used_variables 
                    ),
                    equality_literal
                ],
            )
        case ForallCondition(variables, subcondition):
            subcondition_id, sub_used_variables = _add_condition_to_asp_part(
                subcondition,
                part,
                rule_id_allocator,
                variable_id_allocator,
                object_id_allocator,
                predicate_id_allocator,
                type_id_allocator
            )
            variables_unwrap = [v.value for v in variables]

            used_variables = set(sub_used_variables) - set(variables_unwrap)
            head = make_function_literal(rule_id, used_variables)
            conditions = []

            for quantification_variable in variables:
                type_id = type_id_allocator.get_id_or_insert(quantification_variable.type)
                conditions.append(make_function_literal(
                    type_id,
                    [quantification_variable.value]
                ))

            part.add_rule(
                head,
                [
                    *(
                        make_function_literal(ASP_TRUE, [variable])
                        for variable in used_variables
                    ),
                    part.create_conditional_literal(
                        make_function_literal(subcondition_id, sub_used_variables),
                        conditions
                    )
                ]
            )

    return rule_id, used_variables


def action_definition_asp_part(
    problem: Problem,
    action_definition: ActionDefinition,
    variable_id_allocator: IDAllocator[Variable],
    object_id_allocator: IDAllocator[Object],
    predicate_id_allocator: IDAllocator[Identifier],
    type_id_allocator: IDAllocator[Type],
) -> ASPPart:
    part = ASPPart(ASPPartKind.ACTION_DEFINITION)

    precondition_id, arguments = _add_condition_to_asp_part(
        action_definition.precondition,
        part,
        IDAllocator.from_id_constructor(RuleID),
        variable_id_allocator,
        object_id_allocator,
        predicate_id_allocator,
        type_id_allocator
    )

    body = []
    # Specify the parameters, and their types.
    for parameter in action_definition.parameters:
        variable_id = variable_id_allocator.get_id_or_insert(parameter.value)
        type_id = type_id_allocator.get_id_or_insert(parameter.type)
        body.append(part.create_function_literal(
            str(type_id),
            [part.create_variable(str(variable_id))]
        ))

    body.append(part.create_function_literal(
        str(precondition_id),
        [
            part.create_variable(str(variable_id_allocator.get_id_or_insert(arg)))
            for arg in arguments
        ]
    ))
    head = part.create_function_literal(
        # Parameters is an iterable of Typed items, which have type and .value.
        ASP_ACTION_LABEL, 
        [
            part.create_variable(
                str(variable_id_allocator.get_id_or_insert(parameter.value))
            )
            for parameter in action_definition.parameters
        ]
    )
    part.add_rule(head, body)
    part.add_show_signature(ASP_ACTION_LABEL, len(action_definition.parameters))
    return part
