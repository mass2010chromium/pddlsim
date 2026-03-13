import importlib.resources
from collections.abc import Set
from dataclasses import dataclass
from importlib.abc import Traversable

import pytest

from pddlsim.ast import Domain, GroundedAction, Identifier, Object, Problem
from pddlsim.parser import (
    parse_domain_problem_pair,
)
from pddlsim.simulation import Simulation
from tests import preprocess_traversables

RESOURCES = importlib.resources.files(__name__)


@dataclass
class _GetGroundedActionsCase:
    domain: Domain
    problem: Problem
    expected_grounded_actions: Set[GroundedAction]


def _grounded_action_from_text(text: str) -> GroundedAction:
    text_without_parentheses = text[1:-1]
    elements = text_without_parentheses.split(" ")

    action_name = elements[0]
    grounding = elements[1:]

    return GroundedAction(
        Identifier(action_name), tuple(Object(object_) for object_ in grounding)
    )


def _preprocess_get_grounded_actions_case(
    traversable: Traversable,
) -> _GetGroundedActionsCase:
    domain_text = traversable.joinpath("domain.pddl").read_text()
    problem_text = traversable.joinpath("problem.pddl").read_text()

    domain, problem = parse_domain_problem_pair(domain_text, problem_text)

    expected_grounded_actions = set(
        _grounded_action_from_text(line)
        for line in traversable.joinpath("output.txt").read_text().splitlines()
    )

    return _GetGroundedActionsCase(domain, problem, expected_grounded_actions)


_CASES = preprocess_traversables(
    RESOURCES.joinpath("cases"), _preprocess_get_grounded_actions_case
)


@pytest.mark.parametrize(
    "case",
    _CASES.values(),
    ids=_CASES.keys(),
)
def test_get_grounded_actions(case: _GetGroundedActionsCase) -> None:
    grounded_actions = Simulation.from_domain_and_problem(
        case.domain, case.problem
    ).get_grounded_actions()

    ret = list(grounded_actions)
    print("got (list)", ret)

    grounded_actions = Simulation.from_domain_and_problem(
        case.domain, case.problem
    ).get_grounded_actions()
    ret = set(grounded_actions)
    print("got (set)", ret)

    assert case.expected_grounded_actions == ret
