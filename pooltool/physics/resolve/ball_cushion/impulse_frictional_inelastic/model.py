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
from pooltool.physics.resolve.sphere_half_space_collision import (
    resolve_sphere_half_space_collision,
)

# FIXME: duplicated from han_2005/model.py
Cushion = TypeVar("Cushion", LinearCushionSegment, CircularCushionSegment)


def _solve(ball: Ball, cushion: Cushion) -> Tuple[Ball, Cushion]:
    rvw = resolve_sphere_half_space_collision(
        normal=cushion.get_normal_3d(ball.xyz),
        rvw=ball.state.rvw,
        R=ball.params.R,
        mu_k=ball.params.f_c,
        e=ball.params.e_c,
    )

    # FIXME-3D: add z-velocity back in
    rvw[1][2] = 0.0

    ball.state = BallState(rvw, const.sliding)

    return ball, cushion


@attrs.define
class ImpulseFrictionalInelasticLinear(CoreBallLCushionCollision):
    model: BallLCushionModel = attrs.field(
        default=BallLCushionModel.IMPULSE_FRICTIONAL_INELASTIC, init=False, repr=False
    )

    def solve(
        self, ball: Ball, cushion: LinearCushionSegment
    ) -> Tuple[Ball, LinearCushionSegment]:
        return _solve(ball, cushion)


@attrs.define
class ImpulseFrictionalInelasticCircular(CoreBallCCushionCollision):
    model: BallCCushionModel = attrs.field(
        default=BallCCushionModel.IMPULSE_FRICTIONAL_INELASTIC, init=False, repr=False
    )

    def solve(
        self, ball: Ball, cushion: CircularCushionSegment
    ) -> Tuple[Ball, CircularCushionSegment]:
        return _solve(ball, cushion)
