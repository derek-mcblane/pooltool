from typing import Tuple, TypeVar

import attrs

import pooltool.constants as const
from pooltool.objects.ball.datatypes import Ball, BallState
from pooltool.objects.table.components import (
    CircularCushionSegment,
    LinearCushionSegment,
)
from pooltool.physics.resolve.ball_cushion.core import (
    CoreBallCCushionCollision,
    CoreBallLCushionCollision,
)
from pooltool.physics.resolve.models import BallCCushionModel, BallLCushionModel
from pooltool.physics.resolve.sphere_half_space_collision_IFR import (
    resolve_sphere_half_space_collision_IFR,
)

# FIXME: duplicated from han_2005/model.py
Cushion = TypeVar("Cushion", LinearCushionSegment, CircularCushionSegment)


def _solve(ball: Ball, cushion: Cushion, mu_k, e_n, e_t) -> Tuple[Ball, Cushion]:
    rvw = resolve_sphere_half_space_collision_IFR(
        normal=cushion.get_normal_3d(ball.xyz),
        rvw=ball.state.rvw,
        R=ball.params.R,
        mu_k=mu_k,
        e_n=e_n,
        e_t=e_t,
    )

    # FIXME-3D: add z-velocity back in
    rvw[1][2] = 0.0

    ball.state = BallState(rvw, const.sliding)

    return ball, cushion


@attrs.define
class ImpulseIFRLinear(CoreBallLCushionCollision):
    mu_k: float = attrs.field(default=0.20)
    e_n: float = attrs.field(default=0.85)
    e_t: float = attrs.field(default=-0.50)

    model: BallLCushionModel = attrs.field(
        default=BallLCushionModel.IMPULSE_IFR, init=False, repr=False
    )

    def solve(
        self, ball: Ball, cushion: LinearCushionSegment
    ) -> Tuple[Ball, LinearCushionSegment]:
        return _solve(ball, cushion, self.mu_k, self.e_n, self.e_t)


@attrs.define
class ImpulseIFRCircular(CoreBallCCushionCollision):
    e_n: float = attrs.field(default=0.85)
    e_t: float = attrs.field(default=-0.50)
    mu_k: float = attrs.field(default=0.20)

    model: BallCCushionModel = attrs.field(
        default=BallCCushionModel.IMPULSE_IFR, init=False, repr=False
    )

    def solve(
        self, ball: Ball, cushion: CircularCushionSegment
    ) -> Tuple[Ball, CircularCushionSegment]:
        return _solve(ball, cushion, self.mu_k, self.e_n, self.e_t)
