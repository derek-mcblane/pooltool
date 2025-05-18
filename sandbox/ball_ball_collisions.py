#! /usr/bin/env python

import math
import functools

import attrs

import matplotlib.pyplot as plt
import numpy as np

import pooltool.ptmath as ptmath

from pooltool.objects.ball.datatypes import Ball, BallParams
from pooltool.physics.resolve.ball_ball.friction import (
    AlciatoreBallBallFriction,
    AverageBallBallFriction,
)
from pooltool.physics.resolve.ball_ball.core import BallBallCollisionStrategy
from pooltool.physics.resolve.ball_ball.frictional_inelastic import FrictionalInelastic
from pooltool.physics.resolve.ball_ball.frictional_mathavan import FrictionalMathavan
from pooltool.physics.resolve.ball_ball.frictionless_elastic import FrictionlessElastic

@attrs.define(frozen=True)
class BallBallCollisionExperimentConfig:
    model: BallBallCollisionStrategy
    params: BallParams
    xy_line_of_centers_angle_radians: float = 0.0


@attrs.define
class BallBallCollisionExperiment:
    config: BallBallCollisionExperimentConfig

    ob_i: Ball = attrs.field(init=False)
    @ob_i.default
    def __default_ob(self):
        ob_i = Ball(id="object", params=self.config.params)
        ob_i.state.rvw[0] = np.array([0, 0, self.config.params.R])
        return ob_i

    cb_i: Ball = attrs.field(init=False)
    @cb_i.default
    def __default_cb(self):
        cb_i = Ball(id="cue", params=self.config.params)
        cb_i.set_position_next_to_at_xy_angle(self.ob_i, math.pi + self.config.xy_line_of_centers_angle_radians)
        return cb_i

    @property
    def cb_i_natural_roll(self, cb_speed):
        return ptmath.natural_roll(cb_speed, self.cb_i.params.R)
        
    def result(self, cb_speed: float, cb_topspin: float = 0.0, cb_sidespin: float = 0.0, cut_angle: float = 0.0):
        self.cb_i.setup_motion(cb_speed, self.config.xy_line_of_centers_angle_radians + cut_angle, cb_topspin, cb_sidespin)
        return self.config.model.resolve(self.cb_i, self.ob_i, inplace=False)


def loc_aligned_vector(vector: np.typing.NDArray[np.float64], xy_line_of_centers_angle_radians: float) -> np.typing.NDArray[np.float64]:
    return ptmath.coordinate_rotation(vector, -xy_line_of_centers_angle_radians)


def collision_results_at_varying_cut_angle(experiment_config: BallBallCollisionExperimentConfig, cut_angles, speeds, topspins = None, sidespins = None, spin_as_factor_of_natural_roll = True):
    if topspins is None:
        topspins = [0.0]
    if sidespins is None:
        sidespins = [0.0]

    n_cut_angles = np.size(cut_angles)

    results = {}
    collision_experiment = BallBallCollisionExperiment(experiment_config)

    for speed in speeds:
        natural_roll = ptmath.natural_roll(speed, collision_experiment.cb_i.params.R)
        for topspin in topspins:
            if spin_as_factor_of_natural_roll:
                topspin = topspin * natural_roll
            for sidespin in sidespins:
                if spin_as_factor_of_natural_roll:
                    sidespin = sidespin * natural_roll
                induced_vel = np.empty((n_cut_angles, 3))
                induced_spin = np.empty((n_cut_angles, 3))
                throw_angles = np.empty(n_cut_angles)
                for i, cut_angle in enumerate(cut_angles):
                    cb_f, ob_f = collision_experiment.result(speed, topspin, sidespin, cut_angle)
                    induced_vel[i] = loc_aligned_vector(ob_f.vel, config.xy_line_of_centers_angle_radians)
                    induced_spin[i] = loc_aligned_vector(ob_f.avel, config.xy_line_of_centers_angle_radians)
                    throw_angles[i] = np.atan2(induced_vel[i, 1], induced_vel[i, 0])
                results[(speed, topspin, sidespin)] = (induced_vel, induced_spin, throw_angles)

    return results


def collision_results_at_varying_sidespin(experiment_config: BallBallCollisionExperimentConfig, sidespins, speeds, topspins = None, spin_as_factor_of_natural_roll = True, cut_angles = None):
    if topspins is None:
        topspins = [0.0]
    if cut_angles is None:
        cut_angles = [0.0]

    n_sidespins = np.size(sidespins)

    results = {}
    collision_experiment = BallBallCollisionExperiment(experiment_config)

    for speed in speeds:
        natural_roll = ptmath.natural_roll(speed, collision_experiment.cb_i.params.R)
        for topspin in topspins:
            if spin_as_factor_of_natural_roll:
                topspin = topspin * natural_roll
            for cut_angle in cut_angles:
                induced_vel = np.empty((n_sidespins, 3))
                induced_spin = np.empty((n_sidespins, 3))
                throw_angles = np.empty(n_sidespins)
                for i, sidespin in enumerate(sidespins):
                    if spin_as_factor_of_natural_roll:
                        sidespin = sidespin * natural_roll
                    cb_f, ob_f = collision_experiment.result(speed, topspin, sidespin, cut_angle)
                    induced_vel[i] = loc_aligned_vector(ob_f.vel, config.xy_line_of_centers_angle_radians)
                    induced_spin[i] = loc_aligned_vector(ob_f.avel, config.xy_line_of_centers_angle_radians)
                    throw_angles[i] = np.atan2(induced_vel[i, 1], induced_vel[i, 0])
                results[(speed, topspin, cut_angle)] = (induced_vel, induced_spin, throw_angles)

    return results


def natural_roll_shot_multiple_speeds(config, speeds):
    cut_angles = np.linspace(0, np.pi / 2, 90 * 2, endpoint=False)
    results = collision_results_at_varying_cut_angle(config, cut_angles, speeds, topspins=[1.0], spin_as_factor_of_natural_roll=True);
    cut_angles_deg = np.rad2deg(cut_angles)
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(
        xlabel="cut_angle (deg)",
        ylabel="throw angle (deg)",
        title=f"Natural-Roll Shot Collision at Various Speeds - Throw Angle vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    ax2.set(
        xlabel="cut_angle (deg)",
        ylabel="corkscrew spin / CB natural roll",
        title=f"Natural-Roll-Shot Collision at Various Speeds - Induced OB Corkscrew Spin vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot()
    ax3.set(
        xlabel="cut_angle (deg)",
        ylabel="topspin / CB natural roll",
        title=f"Natural-Roll-Shot Collision at Various Speeds - Induced OB Topspin vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot()
    ax4.set(
        xlabel="cut_angle (deg)",
        ylabel="sidespin / CB natural roll",
        title=f"Natural-Roll-Shot Collision at Various Speeds - Induced OB Sidespin vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )
    for ((speed, _, _), (ob_vels, ob_avels, throw_angles)) in results.items():
        ax.plot(
            cut_angles_deg,
            np.rad2deg(throw_angles),
            label=f"speed={speed:.3} m/s",
        )
        ax2.plot(
            cut_angles_deg,
            ob_avels[:,0],
            label=f"speed={speed:.3} m/s",
        )
        ax3.plot(
            cut_angles_deg,
            ob_avels[:,1],
            label=f"speed={speed:.3} m/s",
        )
        ax4.plot(
            cut_angles_deg,
            ob_avels[:,2],
            label=f"speed={speed:.3} m/s",
        )
    ax.legend()
    ax.grid()
    ax2.legend()
    ax2.grid()
    ax3.legend()
    ax3.grid()
    ax4.legend()
    ax4.grid()
    plt.show()


def stun_shot_multiple_speeds(config, speeds):
    cut_angles = np.linspace(0, np.pi / 2, 90 * 2, endpoint=False)
    results = collision_results_at_varying_cut_angle(config, cut_angles, speeds);
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(
        xlabel="cut_angle (deg)",
        ylabel="throw angle (deg)",
        title=f"Stun Shot Collision at Various Speeds - Throw Angle vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )
    for ((speed, _, _), (_, _, throw_angles)) in results.items():
        ax.plot(
            np.rad2deg(cut_angles),
            np.rad2deg(throw_angles),
            label=f"speed={speed:.3} m/s",
        )
    ax.legend()
    ax.grid()
    plt.show()


def shot_multiple_topspins(config, speed, topspin_factors):
    cut_angles = np.linspace(0, np.pi / 2, 90 * 2, endpoint=False)
    results = collision_results_at_varying_cut_angle(config, cut_angles, [speed], topspin_factors);
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(
        xlabel="cut_angle (deg)",
        ylabel="throw angle (deg)",
        title=f"Head-On Collision with Various Amounts of Topspin - Throw Angle vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )
    for ((speed, topspin, _), (_, _, throw_angles)) in results.items():
        natural_roll = ptmath.natural_roll(speed, ball_params.R)
        topspin_factor = topspin / natural_roll
        ax.plot(
            np.rad2deg(cut_angles),
            np.rad2deg(throw_angles),
            label=f"topspin_factor={topspin_factor:.3}",
        )
    ax.legend()
    ax.grid()
    plt.show()


def head_on_sidespin_shot(config, speed, topspin_factors):
    sidespin_factors = np.linspace(-1.25, 1.25, 100 * 2)
    results = collision_results_at_varying_sidespin(config, sidespin_factors, [speed], topspin_factors);
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(
        xlabel="sidespin / natural roll",
        ylabel="throw angle (deg)",
        title=f"Head-On Collision at Various Speeds - Throw Angle vs. Sidespin\n(model={model.model}, model.friction={model.friction})",
    )
    for ((speed, _, _), (_, _, throw_angles)) in results.items():
        ax.plot(
            sidespin_factors,
            np.rad2deg(throw_angles),
            label=f"speed={speed:.3} m/s",
        )
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    ball_params = BallParams.default()
    alciatore_speeds = [0.447, 1.341, 3.129]
    alciatore_friction = AlciatoreBallBallFriction(a=9.951e-3, b=0.108, c=1.088)
    average_friction = AverageBallBallFriction()
    for model in [
        FrictionalInelastic(friction=alciatore_friction),
        # FrictionalMathavan(friction=alciatore_friction),
        # FrictionalInelastic(friction=average_friction),
        # FrictionalMathavan(friction=average_friction),
    ]:
        config = BallBallCollisionExperimentConfig(model=model, params=ball_params, xy_line_of_centers_angle_radians=-123)
        #natural_roll_shot_multiple_speeds(config, alciatore_speeds)
        #stun_shot_multiple_speeds(config, alciatore_speeds)
        #shot_multiple_topspins(config, alciatore_speeds[1], np.linspace(0, 1, 5))
        head_on_sidespin_shot(config, alciatore_speeds[1], np.linspace(0, 1, 5))
