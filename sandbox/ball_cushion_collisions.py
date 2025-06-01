#! /usr/bin/env python
from typing import Tuple
from numpy.typing import NDArray

import math

import attrs
import matplotlib.pyplot as plt
import numpy as np

import pooltool.ptmath as ptmath
from pooltool.objects.ball.datatypes import Ball, BallParams
from pooltool.objects.table.components import LinearCushionSegment
from pooltool.physics.resolve.ball_cushion.core import CoreBallLCushionCollision
from pooltool.physics.resolve.ball_cushion.impulse_frictional_inelastic import (
    ImpulseFrictionalInelasticLinear,
)
from pooltool.physics.resolve.ball_cushion.han_2005 import (
    Han2005Linear,
)
from pooltool.physics.resolve.ball_cushion.mathavan_2010 import (
    Mathavan2010Linear,
)


def natural_roll_spin_rate(ball_speed: float, R: float):
    return ball_speed / R


def gearing_sidespin(ball_speed: float, R: float, cut_angle: float):
    speed_tangent = ball_speed * math.sin(cut_angle)
    return -speed_tangent / R


def gearing_sidespin_factor(cut_angle: float):
    return math.sin(cut_angle)


def cue_strike_spin_rate(impulse_offset: float, ball_speed: float, R: float):
    """From impulse momentum equations"""
    return 2.5 * ball_speed * impulse_offset / R**2


def cue_strike_spin_rate_factor(impulse_offset: float, R: float) -> float:
    """spin_rate / natural_roll"""
    return 2.5 * impulse_offset / R


def cue_strike_spin_rate_factor_fractional_offset(
    impulse_offset_fraction: float,
) -> float:
    """spin_rate / natural_roll"""
    return 2.5 * impulse_offset_fraction


def cue_strike_spin_rate_factor_percent_english(
    percent_english: float, max_english_radius_fraction=0.5
):
    return cue_strike_spin_rate_factor_fractional_offset(
        (percent_english / 100) * max_english_radius_fraction
    )


@attrs.define(frozen=True)
class BallCushionCollisionExperimentConfig:
    model: CoreBallLCushionCollision
    params: BallParams
    xy_line_of_centers_angle_radians: float = 0.0


@attrs.define
class BallCushionCollisionExperiment:
    config: BallCushionCollisionExperimentConfig

    cushion: LinearCushionSegment = attrs.field(init=False)

    @cushion.default
    def __default_cushion(self):
        length = 2.0
        p1 = ptmath.coordinate_rotation(
            np.array([0.5 * length, 0.0, 0.0]),
            np.pi / 2 + self.config.xy_line_of_centers_angle_radians,
        )
        p2 = -p1
        height = 2.0 * self.config.params.R * 0.635
        p1[2] = height
        p2[2] = height
        cushion = LinearCushionSegment(id="dummy", p1=p1, p2=p2)
        return cushion

    cb_i: Ball = attrs.field(init=False)

    @cb_i.default
    def __default_cb(self):
        cb_i = Ball(id="cue", params=self.config.params)
        contact_position = (self.cushion.p2 + self.cushion.p1) / 2.0
        BallCushionCollisionExperiment.place_ball_next_to_position(
            cb_i,
            contact_position,
            angle_radians=self.config.xy_line_of_centers_angle_radians + math.pi,
        )
        return cb_i

    def setup(
        self,
        cb_speed: float,
        cb_topspin: float = 0.0,
        cb_sidespin: float = 0.0,
        cut_angle: float = 0.0,
    ):
        BallCushionCollisionExperiment.setup_ball_motion(
            self.cb_i,
            cb_speed,
            self.config.xy_line_of_centers_angle_radians - cut_angle,
            cb_topspin,
            cb_sidespin,
        )

    def result(self) -> Tuple[Ball, LinearCushionSegment]:
        return self.config.model.resolve(self.cb_i, self.cushion, inplace=False)

    @staticmethod
    def place_ball_next_to_position(
        ball: Ball,
        position: NDArray[np.float64],
        separation: float = 0.0,
        angle_radians: float = 0.0,
    ):
        ball.state.rvw[0] = np.array([position[0], position[1], ball.params.R]) + ptmath.coordinate_rotation(
            np.array([separation + ball.params.R, 0.0, 0.0]), angle_radians
        )

    @staticmethod
    def setup_ball_motion(ball: Ball, speed: float, direction, topspin, sidespin):
        ball.state.rvw[1] = ptmath.coordinate_rotation(
            np.array([speed, 0.0, 0.0]), direction
        )
        ball.state.rvw[2] = ptmath.coordinate_rotation(
            np.array([0.0, topspin, sidespin]), direction
        )


def collision_results_versus_cut_angle(
    config: BallCushionCollisionExperimentConfig,
    cut_angles,
    speeds,
    topspin_factors=None,
    sidespin_factors=None,
):
    if topspin_factors is None:
        topspin_factors = [0.0]
    if sidespin_factors is None:
        sidespin_factors = [0.0]

    n_cut_angles = np.size(cut_angles)

    results = {}
    collision_experiment = BallCushionCollisionExperiment(config)

    for speed in speeds:
        natural_roll = natural_roll_spin_rate(speed, collision_experiment.cb_i.params.R)
        for topspin_factor in topspin_factors:
            topspin = topspin_factor * natural_roll
            for sidespin_factor in sidespin_factors:
                sidespin = sidespin_factor * natural_roll
                vel = np.empty((n_cut_angles, 3))
                avel = np.empty((n_cut_angles, 3))
                outgoing_vel = np.empty((n_cut_angles, 3))
                outgoing_avel = np.empty((n_cut_angles, 3))
                rebound_angle = np.empty(n_cut_angles)
                for i, cut_angle in enumerate(cut_angles):
                    sidespin = sidespin_factor * natural_roll

                    collision_experiment.setup(speed, topspin, sidespin, cut_angle)
                    vel[i] = collision_experiment.cb_i.vel
                    avel[i] = collision_experiment.cb_i.avel

                    cb_f, _ = collision_experiment.result()
                    outgoing_vel[i] = ptmath.coordinate_rotation(
                        cb_f.vel, -config.xy_line_of_centers_angle_radians
                    )
                    outgoing_avel[i] = ptmath.coordinate_rotation(
                        cb_f.avel, -config.xy_line_of_centers_angle_radians
                    )
                    rebound_angle[i] = np.atan(outgoing_vel[i][1] / outgoing_vel[i][0])

                results[(speed, topspin_factor, sidespin_factor)] = (
                    vel,
                    avel,
                    outgoing_vel,
                    outgoing_avel,
                    rebound_angle
                )

    return results


def plot_rebound_angle_vs_incident_angle(
    title: str, config, speeds, topspin_factors=None, sidespin_factors=None
):
    cut_angles = np.linspace(0, np.pi / 2, 90 * 2, endpoint=False)
    cut_angles_deg = np.rad2deg(cut_angles)

    results = collision_results_versus_cut_angle(
        config, cut_angles, speeds, topspin_factors, sidespin_factors
    )

    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(
        xlabel="incident angle (deg)",
        ylabel="rebound angle (deg)",
        title=title,
    )
    for (speed, topspin_factor, sidespin_factor), (
        _,
        _,
        _,
        _,
        rebound_angles, 
    ) in results.items():
        label = f"speed={speed:.3} m/s"
        if topspin_factors is not None:
            label += f", topspin_factor={topspin_factor:.2}"
        if sidespin_factors is not None:
            label += f", sidespin_factor={sidespin_factor:.2}"
        rebound_angles_deg = np.rad2deg(rebound_angles)
        ax.plot(cut_angles_deg, rebound_angles_deg, label=label)
    ax.legend()
    ax.grid()
    plt.show()


def main():
    ball_params = BallParams.default()
    slow_speed = 0.447
    medium_speed = 1.341
    fast_speed = 3.129
    alciatore_speeds = [slow_speed, medium_speed, fast_speed]
    for model in [ImpulseFrictionalInelasticLinear(), Han2005Linear(), Mathavan2010Linear()]:
        config = BallCushionCollisionExperimentConfig(
            model=model, params=ball_params, xy_line_of_centers_angle_radians=0
        )
        plot_rebound_angle_vs_incident_angle(
            f"Stun-Shot Collision At Various Speeds\nRebound Angle vs. Incident Angle\n(model={config.model.model})",
            config,
            alciatore_speeds,
        )


if __name__ == "__main__":
    main()
