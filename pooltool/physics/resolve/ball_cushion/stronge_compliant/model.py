from typing import Tuple

import attrs
import numpy as np

import pooltool.constants as const
import pooltool.ptmath as ptmath
from pooltool.objects.ball.datatypes import Ball, BallState
from pooltool.objects.table.components import (
    CircularCushionSegment,
    Cushion,
    LinearCushionSegment,
)
from pooltool.physics.resolve.ball_cushion.core import (
    CoreBallCCushionCollision,
    CoreBallLCushionCollision,
)
from pooltool.physics.resolve.models import BallCCushionModel, BallLCushionModel
from pooltool.physics.resolve.stronge_compliant import (
    resolve_collinear_compliant_frictional_inelastic_collision,
)


def _solve(ball: Ball, cushion: Cushion) -> Tuple[Ball, Cushion]:
    rvw = ball.state.rvw.copy()

    # Ensure the normal is pointing in the same direction as the ball's velocity.
    normal_direction = cushion.get_normal_xy(ball.state.rvw)
    if np.dot(normal_direction, rvw[1]) > 0:
        normal_direction = -normal_direction

    relative_contact_velocity = ptmath.surface_velocity(
        rvw, normal_direction, ball.params.R
    )

    v_n_0 = np.dot(normal_direction, relative_contact_velocity)
    n_cross_v = ptmath.cross(normal_direction, relative_contact_velocity)
    v_t_0 = -ptmath.norm3d(n_cross_v)
    tangent_direction = (
        ptmath.cross(n_cross_v, normal_direction) / v_t_0 if v_t_0 != 0.0 else None
    )

    # inverse inertia matrix coefficients for sphere half-space collision
    effective_mass = ball.params.m
    beta_t = 3.5
    beta_n = 1.0

    v_t_f, v_n_f = resolve_collinear_compliant_frictional_inelastic_collision(
        v_t_0=v_t_0,
        v_n_0=v_n_0,
        m=effective_mass,
        beta_t=beta_t,
        beta_n=beta_n,
        mu=ball.params.f_c,
        e_n=ball.params.e_c,
        k_n=1e6,  # TODO: cushion params
        eta_squared=1.1,
    )

    Dv_n = (v_n_f - v_n_0) / beta_n
    rvw[1] += Dv_n * normal_direction
    if tangent_direction is not None:
        Dv_t = (v_t_f - v_t_0) / beta_t
        rvw[1] += Dv_t * tangent_direction
        rvw[2] += (
            Dv_t
            * (2.5 / ball.params.R)
            * ptmath.cross(-normal_direction, tangent_direction)
        )

    # FIXME-3D: add z-velocity back in
    rvw[1][2] = 0.0

    ball.state = BallState(rvw, const.sliding)

    return ball, cushion


@attrs.define
class StrongeCompliantLinear(CoreBallLCushionCollision):
    model: BallLCushionModel = attrs.field(
        default=BallLCushionModel.STRONGE_COMPLIANT, init=False, repr=False
    )

    def solve(
        self, ball: Ball, cushion: LinearCushionSegment
    ) -> Tuple[Ball, LinearCushionSegment]:
        return _solve(ball, cushion)


@attrs.define
class StrongeCompliantCircular(CoreBallCCushionCollision):
    model: BallCCushionModel = attrs.field(
        default=BallCCushionModel.STRONGE_COMPLIANT, init=False, repr=False
    )

    def solve(
        self, ball: Ball, cushion: CircularCushionSegment
    ) -> Tuple[Ball, CircularCushionSegment]:
        return _solve(ball, cushion)
