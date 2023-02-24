from __future__ import annotations

from typing import Dict, Optional, Tuple, Type, Union

from attrs import define, evolve, field

from pooltool.objects.ball.datatypes import Ball, BallHistory
from pooltool.objects.cue.datatypes import Cue
from pooltool.objects.datatypes import NullObject
from pooltool.objects.table.components import (
    CircularCushionSegment,
    LinearCushionSegment,
    Pocket,
)
from pooltool.utils import strenum


class EventType(strenum.StrEnum):
    NONE = strenum.auto()
    BALL_BALL = strenum.auto()
    BALL_LINEAR_CUSHION = strenum.auto()
    BALL_CIRCULAR_CUSHION = strenum.auto()
    BALL_POCKET = strenum.auto()
    STICK_BALL = strenum.auto()
    SPINNING_STATIONARY = strenum.auto()
    ROLLING_STATIONARY = strenum.auto()
    ROLLING_SPINNING = strenum.auto()
    SLIDING_ROLLING = strenum.auto()

    def is_collision(self):
        return self in (
            EventType.BALL_BALL,
            EventType.BALL_CIRCULAR_CUSHION,
            EventType.BALL_LINEAR_CUSHION,
            EventType.BALL_POCKET,
            EventType.STICK_BALL,
        )

    def is_transition(self):
        return self in (
            EventType.SPINNING_STATIONARY,
            EventType.ROLLING_STATIONARY,
            EventType.ROLLING_SPINNING,
            EventType.SLIDING_ROLLING,
        )


Object = Union[
    NullObject,
    Cue,
    Ball,
    Pocket,
    LinearCushionSegment,
    CircularCushionSegment,
]


class AgentType(strenum.StrEnum):
    NULL = strenum.auto()
    CUE = strenum.auto()
    BALL = strenum.auto()
    POCKET = strenum.auto()
    LINEAR_CUSHION_SEGMENT = strenum.auto()
    CIRCULAR_CUSHION_SEGMENT = strenum.auto()


_class_to_type: Dict[Type[Object], AgentType] = {
    NullObject: AgentType.NULL,
    Cue: AgentType.CUE,
    Ball: AgentType.BALL,
    Pocket: AgentType.POCKET,
    LinearCushionSegment: AgentType.LINEAR_CUSHION_SEGMENT,
    CircularCushionSegment: AgentType.CIRCULAR_CUSHION_SEGMENT,
}


@define
class Agent:
    id: str
    agent_type: AgentType

    initial: Optional[Object] = field(default=None)
    final: Optional[Object] = field(default=None)

    def set_initial(self, obj: Object) -> None:
        """Set the object's state pre-event"""
        if self.agent_type == AgentType.NULL:
            return

        self.initial = obj.copy()

        if self.agent_type == AgentType.BALL:
            # In this special case, we drop history fields because they are potentially
            # huge
            assert isinstance(self.initial, Ball)
            self.initial.history = BallHistory()
            self.initial.history_cts = BallHistory()

    def get_final(self) -> Optional[Object]:
        """Return a copy of the object post-event"""
        if self.final is None:
            return None

        return self.final.copy()

    def matches(self, obj: Object) -> bool:
        """Returns whether a given object matches the agent

        Returns True if the object is the correct class type and the IDs match
        """
        correct_class = _class_to_type[type(obj)] == self.agent_type
        return correct_class and obj.id == self.id

    @staticmethod
    def from_object(obj: Object) -> Agent:
        return Agent(id=obj.id, agent_type=_class_to_type[type(obj)])

    def copy(self) -> Agent:
        """Create a deepcopy"""
        return evolve(self)


@define
class Event:
    event_type: EventType
    agents: Tuple[Agent, ...]
    time: float

    def __repr__(self):
        agents = [
            (agent.initial.id if agent.initial is not None else None)
            for agent in self.agents
        ]
        lines = [
            f"<{self.__class__.__name__} object at {hex(id(self))}>",
            f" ├── type   : {self.event_type}",
            f" ├── time   : {self.time}",
            f" └── agents : {agents}",
        ]
        return "\n".join(lines) + "\n"

    def copy(self) -> Event:
        """Create a deepcopy"""
        return evolve(self)