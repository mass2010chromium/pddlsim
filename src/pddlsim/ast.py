"""Definitions for AST data types used for representing PDDL definitions."""

import bisect
import itertools
import operator
import re
from abc import ABC, abstractmethod
from collections import ChainMap
from collections.abc import (
    Container,
    Iterable,
    Iterator,
    Mapping,
    Set,
    Sized,
)
from dataclasses import InitVar, dataclass, field
from decimal import Decimal
from enum import StrEnum
from random import Random
from typing import (
    Any,
    ClassVar,
    Self,
    TypedDict,
    override,
)

from koda_validate import (
    StringValidator,
    TypedDictValidator,
    Validator,
)
from lark.lexer import Token

from pddlsim._serde import Serdeable


class Location(ABC):
    """A location of an AST item."""

    @abstractmethod
    def as_str_with_value(self, value: Any) -> str:
        """Display the value with this location as an annotation."""
        raise NotImplementedError


@dataclass(frozen=True)
class FileLocation(Location):
    """A location of an AST item in a file."""

    line: int
    """The line in which the AST item started."""
    column: int
    """The column in which the AST item started."""

    def __post_init__(self) -> None:
        """Verify the line and column numbers are positive."""
        if self.line <= 0:
            raise ValueError(
                f"line number must be positive, is instead {self.line}"
            )

        if self.column <= 0:
            raise ValueError(
                f"column number must be positive, is instead {self.column}"
            )

    @classmethod
    def _from_token(cls, token: Token) -> "FileLocation":
        if not token.line:
            raise ValueError("token must have line information")

        if not token.column:
            raise ValueError("token must have column information")

        return FileLocation(token.line, token.column)

    @override
    def as_str_with_value(self, value: Any) -> str:
        return f"{value} ({self.line}:{self.column})"


@dataclass(frozen=True)
class EmptyLocation(Location):
    """A dummy location for an AST item.

    Useful for assigning locations to AST items generated programmatically.
    """

    @override
    def as_str_with_value(self, value: Any) -> str:
        return str(value)


@dataclass(frozen=True)
class _Locationed(ABC):
    location: Location = field(
        hash=False,
        compare=False,
        default_factory=EmptyLocation,
        kw_only=True,
    )
    """The location of the AST item."""

    @abstractmethod
    def _as_str_without_location(self) -> str:
        raise NotImplementedError

    @override
    def __str__(self) -> str:
        return self.location.as_str_with_value(self._as_str_without_location())


@dataclass(frozen=True)
class _LocationedSet[T](Iterable[T], Container, Sized, _Locationed):
    _items: Set[T] = field(default_factory=set)

    @classmethod
    def from_raw_parts(
        cls,
        items: Iterable[T],
        *,
        location: Location | None = None,
    ) -> Self:
        item_set: set[T] = set()

        result = cls(
            item_set,
            location=location if location else EmptyLocation(),
        )

        for item in items:
            if item in item_set:
                raise ValueError(
                    f"{item} is defined multiple times in {result}"
                )

            item_set.add(item)

        return result

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    @override
    def __contains__(self, item: object) -> bool:
        return item in self._items

    @override
    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class _LocationedList[T](Iterable[T], Sized, _Locationed):
    _items: list[T] = field(default_factory=list)

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    @override
    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class _LocationedItems[T, U, ID](Iterable[T], Container, Sized, _Locationed):
    _items: dict[ID, U] = field(default_factory=dict)

    @classmethod
    def from_raw_parts(
        cls,
        items: Iterable[T],
        **kwargs,
    ) -> Self:
        item_map: dict[ID, U] = dict()
        result = cls(
            item_map,
            **kwargs,
        )

        for item in items:
            name = cls._get_id(item)

            if name in item_map:
                raise ValueError(
                    f"{cls._item_name()} with name {name} is defined multiple times in {result}"  # noqa: E501
                )

            item_map[name] = cls._get_value(item)

        return result

    @classmethod
    @abstractmethod
    def _item_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_id(cls, item: T) -> ID:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_value(cls, item: T) -> U:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _into_item(cls, id: ID, value: U) -> T:
        raise NotImplementedError

    @override
    def __iter__(self) -> Iterator[T]:
        return (self._into_item(id, value) for id, value in self._items.items())

    def __getitem__(self, id: ID) -> U:
        return self._items[id]

    @override
    def __contains__(self, item: object) -> bool:
        return item in self._items

    @override
    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class _LocationedIDedItems[T, ID](_LocationedItems[T, T, ID]):
    @classmethod
    def _get_value(cls, item: T) -> T:
        return item

    @classmethod
    def _into_item(cls, id: ID, value: T) -> T:
        return value


class Requirement(StrEnum):
    """Represents a PDDL requirement (with extensions)."""

    STRIPS = ":strips"
    """Doesn't add any new features and is assumed."""
    TYPING = ":typing"
    """Allows to specify types for objects, and impose types on variables."""
    DISJUNCTIVE_PRECONDITIONS = ":disjunctive-preconditions"
    """Allows usage of `or` in conditions."""
    NEGATIVE_PRECONDITIONS = ":negative-preconditions"
    """Allows usage of `not` in conditions."""
    EQUALITY = ":equality"
    """Allows usage of `=` predicates."""
    PROBABILISTIC_EFFECTS = ":probabilistic-effects"
    """Allows usage of probabilistic effects."""

    FALLIBLE_ACTIONS = ":fallible-actions"
    """Allows specifying [action fallibilities](https://github.com/galk-research/pddlsim/wiki/Fallible-Actions)."""
    REVEALABLES = ":revealables"
    """Allows problems to use [revealables](https://github.com/galk-research/pddlsim/wiki/Revealables)."""
    MULTIPLE_GOALS = ":multiple-goals"
    """Allows problems specify [multiple goals](https://github.com/galk-research/pddlsim/wiki/Multiple-Goals)."""

    @override
    def __str__(self) -> str:
        return f"`{self.value}`"

    @override
    def __repr__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RequirementsSection(_LocationedSet[Requirement]):
    """Represents a set of requirements, specified for domains and problems."""

    @override
    def _as_str_without_location(self) -> str:
        return "requirements section"

    @override
    def __repr__(self) -> str:
        return f"(:requirements {' '.join(map(repr, self._items))})"


@dataclass(frozen=True)
class Identifier(_Locationed, Serdeable[str]):
    """Represents a PDDL identifier."""

    value: str
    """The identifier's text."""

    _IDENTIFIER_REGEX: ClassVar = re.compile(r"[a-zA-Z][a-zA-Z0-9\-_]*")

    def __post_init__(self) -> None:
        """Validate the identifier (e.g., check if starts with letter)."""
        if not self._IDENTIFIER_REGEX.match(self.value):
            raise ValueError(f"{self.value} is not a valid identifier")

    @override
    def serialize(self) -> str:
        return self.value

    @classmethod
    @override
    def _validator(cls) -> Validator[str]:
        return StringValidator()

    @classmethod
    @override
    def _create(cls, value: str) -> "Identifier":
        return cls(value)

    @override
    def _as_str_without_location(self) -> str:
        return f"`{self.value}`"

    @override
    def __repr__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Variable(Identifier):
    """Represents a PDDL variable."""

    @override
    def _as_str_without_location(self) -> str:
        return f"`?{self.value}`"

    @override
    def __repr__(self) -> str:
        return f"?{self.value}"


@dataclass(frozen=True)
class CustomType(Identifier):
    """Represents a user-defineable type in PDDL."""

    @override
    def __repr__(self) -> str:
        return super().__repr__()


@dataclass(frozen=True)
class ObjectType:
    """Represents the type `object` in PDDL. All types are subtypes of it."""

    @override
    def __str__(self) -> str:
        return "`object`"

    @override
    def __repr__(self) -> str:
        return "object"


type Type = CustomType | ObjectType
"""Represents the type of a PDDL object/variable."""


@dataclass(frozen=True)
class Typed[T]:
    """Represents an AST item with a type attached."""

    value: T
    """The AST item."""
    type: Type
    """The attached type."""

    @override
    def __repr__(self) -> str:
        return f"{self.value!r} - {self.type!r}"


@dataclass(frozen=True)
class _LocationedTypedItems[T](_LocationedItems[Typed[T], Type, T]):
    @classmethod
    @override
    def _get_id(cls, item: Typed[T]) -> T:
        return item.value

    @classmethod
    @override
    def _get_value(cls, item: Typed[T]) -> Type:
        return item.type

    @classmethod
    @override
    def _into_item(cls, id: T, value: Type) -> Typed[T]:
        return Typed(id, value)

    def _validate(self, domain: "Domain") -> None:
        for item in self:
            if (
                isinstance(item.type, CustomType)
                and item.type not in domain.types_section
            ):
                raise ValueError(
                    f"{self._item_name()} {item.value} is of undefined type {item.type}"  # noqa: E501
                )


@dataclass(frozen=True)
class TypesSection(_LocationedTypedItems[CustomType]):
    """Represents the `:types` section in PDDL."""

    def supertype(self, custom_type: CustomType) -> Type:
        """Return the immediate supertype of the provided type.

        The passed value is a `CustomType`, as `ObjectType` has no supertype.
        """
        return self._items[custom_type]

    def is_compatible(self, test_type: Type, tester_type: Type) -> bool:
        """Check if `test_type` is a subtype of `tester_type`.

        This check returns `True` for equal types, and so isn't
        a strict subtyping check.
        """
        if test_type == tester_type:
            return True
        elif isinstance(test_type, CustomType):
            return self.is_compatible(self.supertype(test_type), tester_type)
        else:
            return False

    @classmethod
    @override
    def _item_name(cls) -> str:
        return "type"

    @override
    def _as_str_without_location(self) -> str:
        return "types section"

    @override
    def __repr__(self) -> str:
        return f"(:types {' '.join(map(repr, self))})"


@dataclass(frozen=True)
class Parameters(_LocationedTypedItems[Variable]):
    """Represents action and predicate definition parameters in PDDL."""

    definition: Identifier | None = None

    @classmethod
    @override
    def _item_name(cls) -> str:
        return "parameter"

    @override
    def _as_str_without_location(self) -> str:
        return (
            "parameters"
            if not self.definition
            else f"parameters of {self.definition}"
        )


@dataclass(frozen=True)
class PredicateDefinition:
    """Represents a predicate definition (in domain's `:predicates` section)."""

    name: Identifier
    parameters: Parameters

    @classmethod
    def from_raw_parts(
        cls,
        name: Identifier,
        parameters: Parameters,
    ) -> "PredicateDefinition":
        """Construct a `PredicateDefinition` from subnodes returned by a parser."""  # noqa: E501
        return PredicateDefinition(name, parameters)

    def _validate(self, type_hierarchy: TypesSection) -> None:
        for parameter in self.parameters:
            if (
                isinstance(parameter.type, CustomType)
                and parameter.type not in type_hierarchy
            ):
                raise ValueError(
                    f"parameter {parameter.value} of predicate definition {self.name} is of undefined type {parameter.type}"  # noqa: E501
                )

    @override
    def __repr__(self) -> str:
        return f"({self.name!r} {' '.join(map(repr, self.parameters))})"


@dataclass(frozen=True)
class Object(Identifier):
    """Represents the name of an object in PDDL."""

    @override
    def __repr__(self) -> str:
        return self.value


type Argument = Variable | Object


@dataclass(frozen=True)
class AndCondition[A: Argument]:
    """Represents a conjunction (`and`) of PDDL conditions."""

    subconditions: list["Condition[A]"]
    """The subconditions being conjuncted."""

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        for subcondition in self.subconditions:
            subcondition._validate(parameters, objects, domain)

    @override
    def __repr__(self) -> str:
        return f"(and {' '.join(map(repr, self.subconditions))})"


@dataclass(frozen=True)
class OrCondition[A: Argument](_Locationed):
    """Represents a disjunction (`or`) of PDDL conditions."""

    subconditions: list["Condition[A]"]
    """The subconditions being disjuncted."""

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        if (
            Requirement.DISJUNCTIVE_PRECONDITIONS
            not in domain.requirements_section
        ):
            raise ValueError(
                f"{self} used in condition, but {Requirement.DISJUNCTIVE_PRECONDITIONS} is not in {domain.requirements_section}"  # noqa: E501
            )

        for subcondition in self.subconditions:
            subcondition._validate(parameters, objects, domain)

    @override
    def _as_str_without_location(self) -> str:
        return "disjunction"

    @override
    def __repr__(self) -> str:
        return f"(or {' '.join(map(repr, self.subconditions))})"


@dataclass(frozen=True)
class NotCondition[A: Argument](_Locationed):
    """Represents the negation (`not`) of a PDDL condition."""

    base_condition: "Condition[A]"
    """The base condition being negated."""

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        if (
            Requirement.NEGATIVE_PRECONDITIONS
            not in domain.requirements_section
        ):
            raise ValueError(
                f"{self} used in condition, but {Requirement.NEGATIVE_PRECONDITIONS} is not in {domain.requirements_section}"  # noqa: E501
            )

        self.base_condition._validate(parameters, objects, domain)

    @override
    def _as_str_without_location(self) -> str:
        return "negation"

    @override
    def __repr__(self) -> str:
        return f"(not {self.base_condition!r})"


@dataclass(frozen=True)
class EqualityCondition[A: Argument](_Locationed):
    """Represents an equality predicate (`=`) in PDDL."""

    left_side: A
    """Left side of the equality."""
    right_side: A
    """Right side of the equality."""

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        if Requirement.EQUALITY not in domain.requirements_section:
            raise ValueError(
                f"{self} used in condition, but {Requirement.EQUALITY} is not in {domain.requirements_section}"  # noqa: E501
            )

        def validate_argument(argument: A) -> None:
            match argument:
                case Variable():
                    if argument not in parameters:
                        raise ValueError(
                            f"variable {argument} in {self} is undefined"
                        )
                case Object():
                    if argument not in objects:
                        raise ValueError(
                            f"object {argument} in {self} is undefined"
                        )

        validate_argument(self.left_side)
        validate_argument(self.right_side)

    @override
    def _as_str_without_location(self) -> str:
        return "equality predicate"

    @override
    def __repr__(self) -> str:
        return f"(= {self.left_side!r} {self.right_side!r})"

@dataclass(frozen=True)
class ForallCondition[A: Argument]:
    """Represents a forall predicate (`forall`) in PDDL."""

    variables: list[Typed[Variable]]
    """Parameters local to the forall predicate."""
    condition: "Condition[A]"
    """Single subcondition to evaluate for valid choices of parameters."""

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        scope_dict = dict(parameters._items)
        for typed_variable in self.variables:
            scope_dict[typed_variable.value] = typed_variable.type
        scoped_parameters = Parameters(_items=scope_dict, definition=parameters.definition)
        self.condition._validate(scoped_parameters, objects, domain)

    @override
    def _as_str_without_location(self) -> str:
        return "forall predicate"

    @override
    def __repr__(self) -> str:
        parameters = (
            f"({' '.join(map(repr, self.variables))})"
        )
        return f"(forall {parameters} {self.condition!r})"


class _SerializedPredicate(TypedDict):
    name: Any
    assignment: list[Any]


@dataclass(frozen=True)
class Predicate[A: Argument](Serdeable[_SerializedPredicate]):
    """Represents the instantiation of a predicate in PDDL."""

    name: Identifier
    """The name of the predicate being instantiated."""
    assignment: tuple[A, ...]
    """The instantiation of the predicate."""

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        if self.name not in domain.predicates_section:
            raise ValueError(f"predicate {self.name} is undefined")

        predicate_definition = domain.predicates_section[self.name]

        found_arity = len(self.assignment)
        expected_arity = len(predicate_definition.parameters)

        if found_arity != expected_arity:
            raise ValueError(
                f"predicate {self.name} is defined with arity {expected_arity}, but is used with arity {found_arity}"  # noqa: E501
            )

        for parameter, argument in zip(
            predicate_definition.parameters, self.assignment, strict=True
        ):
            match argument:
                case Variable():
                    if argument not in parameters:
                        raise ValueError(
                            f"variable {argument} in {self.name} is undefined"
                        )

                    argument_type = parameters[argument]  # type: ignore
                case Object():
                    if argument not in objects:
                        raise ValueError(
                            f"object {argument} in {self.name} is undefined"
                        )

                    argument_type = objects[argument]  # type: ignore

            if not domain.types_section.is_compatible(
                argument_type, parameter.type
            ):
                raise ValueError(
                    f"argument {argument} in {self.name} is of type {argument_type}, but is supposed to be of type {parameter.type}"  # noqa: E501
                )

    @override
    def serialize(self) -> _SerializedPredicate:
        return _SerializedPredicate(
            name=self.name.serialize(),
            assignment=[argument.serialize() for argument in self.assignment],
        )

    @classmethod
    @override
    def _validator(cls) -> Validator[_SerializedPredicate]:
        return TypedDictValidator(_SerializedPredicate)

    @classmethod
    @override
    def _create(cls, value: _SerializedPredicate) -> "Predicate[A]":
        return Predicate(
            Identifier.deserialize(value["name"]),
            tuple(
                Object.deserialize(argument)  # type: ignore
                for argument in value["assignment"]
            ),
        )

    @override
    def __repr__(self) -> str:
        return f"({self.name!r} {' '.join(map(repr, self.assignment))})"

    @override
    def __str__(self):
        return f"`{self!r}`"


type Condition[A: Argument] = (
    AndCondition[A]
    | OrCondition[A]
    | NotCondition[A]
    | EqualityCondition[A]
    | ForallCondition[A]
    | Predicate[A]
)
"""Represents a condition in PDDL, over objects, or variables."""


@dataclass(frozen=True)
class NotPredicate[A: Argument]:
    """Represents the effect of removing a predicate from the state in PDDL."""

    base_predicate: Predicate[A]
    """The predicate to remove."""

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        self.base_predicate._validate(parameters, objects, domain)

    @override
    def __repr__(self) -> str:
        return f"(not {self.base_predicate!r})"


type Atom[A: Argument] = Predicate[A] | NotPredicate[A]
"""Represents an effect adding/removing a predicate from the state in PDDL."""


@dataclass(frozen=True)
class AndEffect[A: Argument]:
    """Represents an effect of applying all subeffects to the state in PDDL."""

    subeffects: list["Effect[A]"] = field(default_factory=list)
    """The subeffects to apply to the state."""

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        for subeffect in self.subeffects:
            subeffect._validate(parameters, objects, domain)

    @override
    def __repr__(self) -> str:
        return f"(and {' '.join(map(repr, self.subeffects))})"


@dataclass(frozen=True)
class ProbabilisticEffect[A: Argument](Iterable, _Locationed):
    """Represents a probabilistic effect, choosing a subeffect at random.

    The primary constructor for this class is
    `ProbabilisticEffect.from_possibilities`.
    """

    _possible_effects: list["Effect[A]"]
    _cummulative_probabilities: list[Decimal]

    @classmethod
    def from_possibilities(
        cls,
        possibilities: list[tuple[Decimal, "Effect[A]"]],
        *,
        location: Location | None = None,
    ) -> "ProbabilisticEffect":
        """Construct a `ProbabilisticEffect` from effect-probability pairs."""
        possible_effects: list[Effect[A]] = [
            effect for _, effect in possibilities
        ]
        cummulative_probabilities: list[Decimal] = list(
            itertools.accumulate(
                (probability for probability, _ in possibilities),
                operator.add,
                initial=Decimal(),
            )
        )

        result = ProbabilisticEffect(
            possible_effects,
            cummulative_probabilities,
            location=location if location else EmptyLocation(),
        )

        cummulative_probability = cummulative_probabilities[-1]

        if cummulative_probability > 1:
            raise ValueError(
                f"total probability of {result} mustn't be greater than 1, is {cummulative_probability}"  # noqa: E501
            )

        return result

    def choose_possibility(self, rng: Random | None = None) -> "Effect[A]":
        """Choose an effect according to their probabilities.

        Optionally, one can specify the RNG to use for the calculations.
        """
        index = bisect.bisect(
            self._cummulative_probabilities, (rng if rng else Random()).random()
        )

        if index == len(self._possible_effects):
            return AndEffect()  # Empty effect

        return self._possible_effects[index]

    def _validate(
        self,
        parameters: Parameters,
        objects: Mapping[Object, Type],
        domain: "Domain",
    ) -> None:
        if Requirement.PROBABILISTIC_EFFECTS not in domain.requirements_section:
            raise ValueError(
                f"{self} used in action, but {Requirement.PROBABILISTIC_EFFECTS} does not appear in {domain.requirements_section}"  # noqa: E501
            )

        for subeffect in self._possible_effects:
            subeffect._validate(parameters, objects, domain)

    @override
    def _as_str_without_location(self) -> str:
        return "probabilistic effect"

    @override
    def __iter__(self) -> Iterator[tuple[Decimal, "Effect[A]"]]:
        probabilities = itertools.chain(
            (self._cummulative_probabilities[0],),
            (
                b - a
                for a, b in itertools.pairwise(self._cummulative_probabilities)
            ),
        )

        return zip(probabilities, self._possible_effects, strict=True)

    @override
    def __repr__(self) -> str:
        def repr_subeffect(
            pair: tuple[Decimal, "Effect[A]"],
        ) -> str:
            return f"{pair[0]} {pair[1]!r}"

        return f"(probabilistic {' '.join(map(repr_subeffect, self))})"


type Effect[A: Argument] = AndEffect[A] | ProbabilisticEffect[A] | Atom[A]
"""Represents a PDDL effect, over objects, or variables."""


@dataclass(frozen=True)
class ActionDefinition:
    """Represents an action definition in PDDL."""

    name: Identifier
    """The name of the action."""
    parameters: Parameters
    """The type of each parameter."""
    precondition: Condition[Argument]
    """The precondition of the action."""
    effect: Effect[Argument]
    """The effect of the action."""

    @classmethod
    def from_raw_parts(
        cls,
        name: Identifier,
        parameters: Parameters,
        precondition: Condition[Argument],
        effect: Effect[Argument],
    ) -> "ActionDefinition":
        """Construct an `ActionDefinition` from subnodes returned from a parser."""  # noqa: E501
        return ActionDefinition(name, parameters, precondition, effect)

    def _validate(self, domain: "Domain") -> None:
        self.parameters._validate(domain)

        self.precondition._validate(
            self.parameters, domain.constants_section._items, domain
        )
        self.effect._validate(
            self.parameters, domain.constants_section._items, domain
        )

    @override
    def __repr__(self) -> str:
        parameters_section = (
            f":parameters ({' '.join(map(repr, self.parameters))})"
        )

        return f"(:action {self.name!r} {parameters_section} :precondition {self.precondition!r} :effect {self.effect!r})"  # noqa: E501


class ConstantsSection(_LocationedTypedItems[Object]):
    """Represents the constants section of a PDDL domain."""

    @classmethod
    @override
    def _item_name(cls) -> str:
        return "constant"

    @override
    def _as_str_without_location(self) -> str:
        return "constants section"

    @override
    def __repr__(self) -> str:
        return f"(:constants {' '.join(map(repr, self))})"


class PredicatesSection(_LocationedIDedItems[PredicateDefinition, Identifier]):
    """Represents the predicates section of a PDDL domain."""

    @classmethod
    @override
    def _get_id(cls, item: PredicateDefinition) -> Identifier:
        return item.name

    @classmethod
    @override
    def _item_name(cls) -> str:
        return "predicate"

    @override
    def _as_str_without_location(self) -> str:
        return "predicates section"

    def _validate(self, domain: "Domain") -> None:
        for predicate_definition in self:
            predicate_definition._validate(domain.types_section)

    @override
    def __repr__(self) -> str:
        return f"(:predicates {' '.join(map(repr, self))})"


class ActionsSection(_LocationedIDedItems[ActionDefinition, Identifier]):
    """Represents the actions section of a PDDL domain."""

    @classmethod
    @override
    def _get_id(cls, item: ActionDefinition) -> Identifier:
        return item.name

    @classmethod
    @override
    def _item_name(cls) -> str:
        return "action"

    @override
    def _as_str_without_location(self) -> str:
        return "actions section"

    def _validate(self, domain: "Domain") -> None:
        for action_definition in self:
            action_definition._validate(domain)

    @override
    def __repr__(self) -> str:
        return " ".join(map(repr, self))


@dataclass(frozen=True)
class Domain:
    """Represents a PDDL domain."""

    name: Identifier
    """The name of the domain."""
    requirements_section: RequirementsSection
    """The domain's requirements"""
    types_section: TypesSection
    """The domain's type hierarchy."""
    constants_section: ConstantsSection
    """The domain's constants, and their types."""
    predicates_section: PredicatesSection
    """The domain's predicate definitions."""
    actions_section: ActionsSection
    """The domain's action definitions."""

    @classmethod
    def from_raw_parts(
        cls,
        name: Identifier,
        requirements_section: RequirementsSection,
        types_section: TypesSection | None,
        constants_section: ConstantsSection,
        predicates_section: PredicatesSection,
        actions_section: ActionsSection,
    ) -> "Domain":
        """Construct a `Domain` from the subnodes returned by a parser."""
        if (
            types_section is not None
            and Requirement.TYPING not in requirements_section
        ):
            raise ValueError(
                f"{types_section} is defined in domain, but {Requirement.TYPING} does not appear in {requirements_section}"  # noqa: E501
            )

        return Domain(
            name,
            requirements_section,
            types_section if types_section else TypesSection(),
            constants_section,
            predicates_section,
            actions_section,
        )

    def __post_init__(self) -> None:
        """Validate the domain (e.g., make sure all referenced types exist)."""
        self._validate()

    def _validate(self) -> None:
        self.constants_section._validate(self)
        self.predicates_section._validate(self)
        self.actions_section._validate(self)

    def as_pddl(self) -> str:
        """Return a PDDL string representing this domain."""
        domain_name = f"(domain {self.name!r})"

        return f"(define {domain_name} {self.requirements_section!r} {self.types_section!r} {self.constants_section!r} {self.predicates_section!r} {self.actions_section!r})"  # noqa: E501


class ObjectsSection(_LocationedTypedItems[Object]):
    """Represents the objects section of a PDDL problem."""

    @classmethod
    @override
    def _item_name(cls) -> str:
        return "object"

    @override
    def _as_str_without_location(self) -> str:
        return "objects section"

    @override
    def __repr__(self) -> str:
        return f"(:objects {' '.join(map(repr, self))})"

    def _validate(self, domain: Domain) -> None:
        super()._validate(domain)

        for object_ in self:
            if object_.value in domain.constants_section:
                raise ValueError(
                    f"object {object_.value} shares name with a constant in domain"  # noqa: E501
                )


@dataclass(frozen=True)
class GroundedActionSchematic[A: Argument]:
    """Stores a partially grounded action: the action name, and grounding.

    Used to represent an action schematic in an `ActionFallibility`.
    These are used to declare which grounded actions the fallibility applies to.
    """

    name: Identifier
    """The name of the action."""
    grounding: tuple[A, ...]
    """The action's grounding."""

    def does_match(self, grounded_action: "GroundedAction") -> bool:
        """Check if the grounded action matches the schematic.

        This means that for parameter of the action, the grounded object
        matches the specified object, or in the case of variables, parameters
        with the same variable, are grounded to the same object.
        """
        variable_assignment: dict[Variable, Object] = {}

        for argument, object_ in zip(
            self.grounding, grounded_action.grounding, strict=True
        ):
            match argument:
                case Variable():
                    if (
                        variable_assignment.setdefault(argument, object_)
                        != object_
                    ):
                        return False
                case Object() if argument != object_:
                    return False

        return True

    def _validate(
        self,
        objects: Mapping[Object, Type],
        domain: Domain,
    ) -> None:
        if self.name not in domain.actions_section:
            raise ValueError(f"action with name {self.name} is undefined")

        action_parameters = domain.actions_section[self.name].parameters

        found_arity = len(self.grounding)
        expected_arity = len(action_parameters)

        if found_arity != expected_arity:
            raise ValueError(
                f"{self.name} is defined with arity {expected_arity}, but is used with arity {found_arity}"  # noqa: E501
            )

        for argument, parameter in zip(
            self.grounding, action_parameters, strict=True
        ):
            match argument:
                case Object():
                    if argument not in objects:
                        raise ValueError(
                            f"object with name {argument} is undefined"
                        )

                    object_type = objects[argument]

                    if object_type != parameter.type:
                        raise ValueError(
                            f"object {argument} in {self.name} is of type {object_type}, but is supposed to be of type {parameter.type}"  # noqa: E501
                        )

    @override
    def __repr__(self) -> str:
        return f"({self.name!r} {' '.join(map(repr, self.grounding))})"


class _SerializedGroundedAction(TypedDict):
    name: Any
    grounding: list[Any]


@dataclass(frozen=True)
class GroundedAction(
    GroundedActionSchematic[Object], Serdeable[_SerializedGroundedAction]
):
    """Stores a grounded action: the action name, and grounding.

    Used to represent actions by the agent that can change the simulation
    (like actions in a PDDL domain). When creating an agent, this is
    the principal value to use to indicate it is performing an action
    in the environment.
    """

    @override
    def serialize(self) -> _SerializedGroundedAction:
        return _SerializedGroundedAction(
            name=self.name.serialize(),
            grounding=[object_.serialize() for object_ in self.grounding],
        )

    @classmethod
    @override
    def _validator(cls) -> Validator[_SerializedGroundedAction]:
        return TypedDictValidator(_SerializedGroundedAction)

    @classmethod
    def _create(cls, value: _SerializedGroundedAction) -> "GroundedAction":
        return GroundedAction(
            Identifier.deserialize(value["name"]),
            tuple(
                Object.deserialize(object_) for object_ in value["grounding"]
            ),
        )

    @override
    def __repr__(self) -> str:
        return super().__repr__()


@dataclass(frozen=True)
class ActionFallibility(_Locationed):
    """Represents an [action fallibility](https://github.com/galk-research/pddlsim/wiki/Fallible-Actions)."""

    grounded_action_schematic: GroundedActionSchematic
    """The name of the action."""
    condition: Condition[Object]
    """The failure condition of the action."""
    with_probability: Decimal = field(
        default=Decimal("1"), compare=False, hash=False
    )
    """The probability of failure on condition satisfaction."""

    def __post_init__(self) -> None:
        """Make sure the specified probability is a valid probability."""
        if not (self.with_probability <= 1):
            raise ValueError(
                f"{self} is with impossible probability {self.with_probability}"
            )

    @override
    def _as_str_without_location(self) -> str:
        return "action fallibility"

    def _validate(self, objects: Mapping[Object, Type], domain: Domain) -> None:
        self.grounded_action_schematic._validate(objects, domain)
        self.condition._validate(Parameters(), objects, domain)

    @override
    def __repr__(self) -> str:
        return f"(:action {self.grounded_action_schematic!r} :on {self.with_probability} {self.condition!r})"  # noqa: E501


@dataclass(frozen=True)
class ActionFallibilitiesSection(_LocationedList[ActionFallibility]):
    """Represents the `:fails` section of a PDDL problem."""

    @override
    def _as_str_without_location(self) -> str:
        return "action fallibilities section"

    def _validate(self, objects: Mapping[Object, Type], domain: Domain) -> None:
        for fallibility in self:
            fallibility._validate(objects, domain)

    @override
    def __repr__(self) -> str:
        return f"(:fails {' '.join(map(repr, self))})"


@dataclass(frozen=True)
class Revealable(_Locationed):
    """Represents a [revealable](https://github.com/galk-research/pddlsim/wiki/Revealables)."""

    effect: Effect[Object]
    """The revealable's effect."""
    condition: Condition[Object]
    """The revealable's activation condition."""
    with_probability: Decimal = field(
        default=Decimal("1"), compare=False, hash=False
    )
    """The revealable's activation probability."""

    def __post_init__(self) -> None:
        """Make sure the probability specified is between 0 and 1."""
        if not (0 <= self.with_probability <= 1):
            raise ValueError(
                f"{self} is with impossible probability {self.with_probability}"
            )

    def _validate(self, objects: Mapping[Object, Type], domain: Domain) -> None:
        self.condition._validate(Parameters(), objects, domain)
        self.effect._validate(Parameters(), objects, domain)

    @override
    def _as_str_without_location(self) -> str:
        return "revealable"

    @override
    def __repr__(self) -> str:
        return (
            f"(when {self.with_probability} {self.condition!r} {self.effect!r})"
        )


@dataclass(frozen=True)
class RevealablesSection(_LocationedList[Revealable]):
    """Represents the revealables section of a PDDL problem."""

    @override
    def _as_str_without_location(self) -> str:
        return "revealables section"

    def _validate(self, objects: Mapping[Object, Type], domain: Domain) -> None:
        for fallibility in self:
            fallibility._validate(objects, domain)

    @override
    def __repr__(self) -> str:
        return f"(:reveals {' '.join(map(repr, self))})"


class InitializationSection(_LocationedSet[Predicate[Object]]):
    """Represents the initialization section of a PDDL problem."""

    @override
    def _as_str_without_location(self) -> str:
        return "initialization section"

    def _validate(self, objects: Mapping[Object, Type], domain: Domain) -> None:
        for predicate in self:
            predicate._validate(Parameters(), objects, domain)

    @override
    def __repr__(self) -> str:
        return f"(:init {' '.join(map(repr, self))})"


@dataclass(frozen=True)
class GoalsSection(_LocationedList[Condition[Object]]):
    """Represents the `:goals`/`:goal` section of a PDDL problem."""

    @override
    def _as_str_without_location(self) -> str:
        return "goals section"

    def _validate(self, objects: Mapping[Object, Type], domain: Domain) -> None:
        for condition in self:
            condition._validate(Parameters(), objects, domain)

    @override
    def __repr__(self) -> str:
        keyword = ":goal" if len(self) == 1 else ":goals"

        return f"({keyword} {' '.join(map(repr, self))})"


@dataclass(frozen=True)
class RawProblem:
    """Represents a PDDL problem, without any validation."""

    name: Identifier
    """The problem's name."""
    used_domain_name: Identifier
    """The name of the domain used by the problem."""
    requirements_section: RequirementsSection
    """The problem's requirements."""
    objects_section: ObjectsSection
    """The objects of the PDDL problem (and their types)."""
    action_fallibilities_section: ActionFallibilitiesSection
    """The action fallibilities of the problem."""
    revealables_section: RevealablesSection
    """The revealables of the problem."""
    initialization_section: InitializationSection
    """The problem's initialization."""
    goals_section: GoalsSection
    """The problem's goal conditions."""

    @classmethod
    def from_raw_parts(
        cls,
        name: Identifier,
        used_domain_name: Identifier,
        requirements_section: RequirementsSection,
        objects_section: ObjectsSection,
        action_fallibilities_section: ActionFallibilitiesSection | None,
        revealables_section: RevealablesSection | None,
        initialization_section: InitializationSection,
        goal: GoalsSection | Condition[Object],
    ) -> "RawProblem":
        """Construct a `RawProblem` from the subnodes returned by a parser."""
        if (
            action_fallibilities_section is not None
            and Requirement.FALLIBLE_ACTIONS not in requirements_section
        ):
            raise ValueError(
                f"{action_fallibilities_section} defined, but {Requirement.FALLIBLE_ACTIONS} does not appear in {requirements_section}"  # noqa: E501
            )

        if (
            revealables_section is not None
            and Requirement.REVEALABLES not in requirements_section
        ):
            raise ValueError(
                f"{revealables_section} defined, but {Requirement.REVEALABLES} does not appear in {requirements_section}"  # noqa: E501
            )

        if (
            isinstance(goal, GoalsSection)
            and Requirement.MULTIPLE_GOALS not in requirements_section
        ):
            raise ValueError(
                f"{goal} defined, but {Requirement.MULTIPLE_GOALS} does not appear in {requirements_section}"  # noqa: E501
            )

        return RawProblem(
            name,
            used_domain_name,
            requirements_section,
            objects_section,
            action_fallibilities_section
            if action_fallibilities_section
            else ActionFallibilitiesSection(),
            revealables_section
            if revealables_section
            else RevealablesSection(),
            initialization_section,
            goal if isinstance(goal, GoalsSection) else GoalsSection([goal]),
        )


@dataclass(frozen=True)
class Problem:
    """Represents a PDDL problem."""

    raw_problem: RawProblem
    """The backing raw problem."""
    domain: InitVar[Domain]
    """The domain used for validation."""

    @property
    def name(self) -> Identifier:
        """The problem's name."""
        return self.raw_problem.name

    @property
    def used_domain_name(self) -> Identifier:
        """The name of the domain used by the problem."""
        return self.raw_problem.used_domain_name

    @property
    def requirements_section(self) -> RequirementsSection:
        """The problem's requirements."""
        return self.raw_problem.requirements_section

    @property
    def objects_section(self) -> ObjectsSection:
        """The objects of the PDDL problem (and their types)."""
        return self.raw_problem.objects_section

    @property
    def action_fallibilities_section(self) -> ActionFallibilitiesSection:
        """The action fallibilities of the problem."""
        return self.raw_problem.action_fallibilities_section

    @property
    def revealables_section(self) -> RevealablesSection:
        """The revealables of the problem."""
        return self.raw_problem.revealables_section

    @property
    def initialization_section(self) -> InitializationSection:
        """The problem's initialization."""
        return self.raw_problem.initialization_section

    @property
    def goals_section(self) -> GoalsSection:
        """The problem's goal conditions."""
        return self.raw_problem.goals_section

    def __post_init__(self, domain: Domain) -> None:
        """Validate the problem (e.g., that all referenced predicates exist)."""
        self._validate(domain)

    def _validate(self, domain: Domain) -> None:
        self._validate_used_domain_name(domain)
        self.objects_section._validate(domain)

        objects = ChainMap(
            self.objects_section._items, domain.constants_section._items
        )

        self.action_fallibilities_section._validate(objects, domain)
        self.initialization_section._validate(objects, domain)
        self.goals_section._validate(objects, domain)

    def _validate_used_domain_name(self, domain: Domain) -> None:
        if domain.name != self.used_domain_name:
            raise ValueError(
                f"used domain name {self.used_domain_name} doesn't match paired domain name"  # noqa: E501
            )

    def as_pddl(
        self,
        show_action_fallibilities: bool = True,
        show_revealables: bool = True,
    ) -> str:
        """Return a PDDL string representing this problem."""
        problem_name = f"(problem {self.name!r})"
        used_domain_name = f"(:domain {self.used_domain_name!r})"

        # Some parsers don't support a problem-specific requirements section,
        # ]so we omit it completely if not necessary.
        requirements_section = (
            repr(self.requirements_section) if self.requirements_section else ""
        )

        action_fallibilities_section = (
            repr(self.action_fallibilities_section)
            if show_action_fallibilities and self.action_fallibilities_section
            else ""
        )
        revealables_section = (
            repr(self.revealables_section)
            if show_revealables and self.revealables_section
            else ""
        )

        return f"(define {problem_name} {used_domain_name} {requirements_section} {self.objects_section!r} {action_fallibilities_section} {revealables_section} {self.initialization_section!r} {self.goals_section!r})"  # noqa: E501
