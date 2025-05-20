#! /usr/bin/env python

import math

import attrs
import matplotlib.pyplot as plt
import numpy as np

import pooltool.ptmath as ptmath
from pooltool.objects.ball.datatypes import Ball, BallParams
from pooltool.physics.resolve.ball_ball.core import BallBallCollisionStrategy
from pooltool.physics.resolve.ball_ball.friction import (
    AlciatoreBallBallFriction,
    AverageBallBallFriction,
)
from pooltool.physics.resolve.ball_ball.frictional_inelastic import FrictionalInelastic
from pooltool.physics.resolve.ball_ball.frictional_mathavan import FrictionalMathavan


def natural_roll_spin_rate(ball_speed: float, R: float):
    return ball_speed / R


def gearing_sidespin(ball_speed: float, R: float, cut_angle: float):
    speed_tangent = ball_speed * math.sin(cut_angle)
    return -speed_tangent / self.params.R


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
        BallBallCollisionExperiment.place_ball_next_to(
            cb_i, self.ob_i, self.config.xy_line_of_centers_angle_radians + math.pi
        )
        return cb_i

    def result(
        self,
        cb_speed: float,
        cb_topspin: float = 0.0,
        cb_sidespin: float = 0.0,
        cut_angle: float = 0.0,
    ):
        BallBallCollisionExperiment.setup_ball_motion(
            self.cb_i,
            cb_speed,
            self.config.xy_line_of_centers_angle_radians - cut_angle,
            cb_topspin,
            cb_sidespin,
        )
        return self.config.model.resolve(self.cb_i, self.ob_i, inplace=False)

    @staticmethod
    def place_ball_next_to(
        ball: Ball, other_ball: Ball, angle: float, separation: float = 0.0
    ):
        ball.state.rvw[0] = other_ball.xyz + ptmath.coordinate_rotation(
            np.array([other_ball.params.R + separation + ball.params.R, 0, 0]), angle
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
    config: BallBallCollisionExperimentConfig,
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
    collision_experiment = BallBallCollisionExperiment(config)

    for speed in speeds:
        natural_roll = natural_roll_spin_rate(speed, collision_experiment.cb_i.params.R)
        for topspin_factor in topspin_factors:
            topspin = topspin_factor * natural_roll
            for sidespin_factor in sidespin_factors:
                sidespin = sidespin_factor * natural_roll
                induced_vel = np.empty((n_cut_angles, 3))
                induced_spin = np.empty((n_cut_angles, 3))
                throw_angles = np.empty(n_cut_angles)
                for i, cut_angle in enumerate(cut_angles):
                    cb_f, ob_f = collision_experiment.result(
                        speed, topspin, sidespin, cut_angle
                    )
                    induced_vel[i] = ptmath.coordinate_rotation(
                        ob_f.vel, -config.xy_line_of_centers_angle_radians
                    )
                    induced_spin[i] = ptmath.coordinate_rotation(
                        ob_f.avel, -config.xy_line_of_centers_angle_radians
                    )
                    throw_angles[i] = -np.atan2(induced_vel[i, 1], induced_vel[i, 0])
                results[(speed, topspin_factor, sidespin_factor)] = (
                    induced_vel,
                    induced_spin,
                    throw_angles,
                )

    return results


def collision_results_versus_sidespin(
    config: BallBallCollisionExperimentConfig,
    sidespin_factors,
    speeds,
    topspin_factors=None,
    cut_angles=None,
):
    if topspin_factors is None:
        topspin_factors = [0.0]
    if cut_angles is None:
        cut_angles = [0.0]

    n_sidespins = np.size(sidespin_factors)

    results = {}
    collision_experiment = BallBallCollisionExperiment(config)

    for speed in speeds:
        natural_roll = natural_roll_spin_rate(speed, collision_experiment.cb_i.params.R)
        for topspin_factor in topspin_factors:
            topspin = topspin_factor * natural_roll
            for cut_angle in cut_angles:
                induced_vel = np.empty((n_sidespins, 3))
                induced_spin = np.empty((n_sidespins, 3))
                throw_angles = np.empty(n_sidespins)
                for i, sidespin_factor in enumerate(sidespin_factors):
                    sidespin = sidespin_factor * natural_roll
                    cb_f, ob_f = collision_experiment.result(
                        speed, topspin, sidespin, cut_angle
                    )
                    induced_vel[i] = ptmath.coordinate_rotation(
                        ob_f.vel, -config.xy_line_of_centers_angle_radians
                    )
                    induced_spin[i] = ptmath.coordinate_rotation(
                        ob_f.avel, -config.xy_line_of_centers_angle_radians
                    )
                    throw_angles[i] = -np.atan2(induced_vel[i, 1], induced_vel[i, 0])
                results[(speed, topspin_factor, cut_angle)] = (
                    induced_vel,
                    induced_spin,
                    throw_angles,
                )

    return results


def plot_throw_vs_cut_angle(
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
        xlabel="cut angle (deg)",
        ylabel="throw angle (deg)",
        title=title,
    )
    for (speed, topspin_factor, sidespin_factor), (
        _,
        _,
        throw_angles,
    ) in results.items():
        label = f"speed={speed:.3} m/s"
        if topspin_factors is not None:
            label += f", topspin_factor={topspin_factor:.2}"
        if sidespin_factors is not None:
            label += f", sidespin_factor={sidespin_factor:.2}"
        ax.plot(cut_angles_deg, np.rad2deg(throw_angles), label=label)
    ax.legend()
    ax.grid()
    plt.show()


def plot_throw_vs_sidespin_factor(
    title: str, config, speeds, topspin_factors=None, cut_angles=None
):
    sidespin_factors = np.linspace(-1.25, 1.25, 125 * 2)

    results = collision_results_versus_sidespin(
        config, sidespin_factors, speeds, topspin_factors, cut_angles
    )

    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(xlabel="sidespin / natural roll", ylabel="throw angle (deg)", title=title)
    for (speed, topspin_factor, cut_angle), (_, _, throw_angles) in results.items():
        label = f"speed={speed:.3} m/s"
        if topspin_factors is not None:
            label += f", topspin_factor={topspin_factor:.2}"
        if cut_angles is not None:
            cut_angle_deg = math.degrees(cut_angle)
            label += f", cut_angle={cut_angle_deg:.3} deg"
        ax.plot(sidespin_factors, np.rad2deg(throw_angles), label=label)
    ax.legend()
    ax.grid()
    plt.show()


def plot_throw_vs_percent_sidespin(
    title: str,
    config,
    speeds,
    topspin_factors=None,
    cut_angles=None,
    max_english_radius_fraction=0.5,
):
    sidespin_percentages = np.linspace(-100, 100, 100 * 2)
    sidespin_factors = np.array(
        [
            cue_strike_spin_rate_factor_percent_english(
                sidespin_percentage, max_english_radius_fraction
            )
            for sidespin_percentage in sidespin_percentages
        ]
    )

    results = collision_results_versus_sidespin(
        config, sidespin_factors, speeds, topspin_factors, cut_angles
    )

    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(xlabel="sidespin (% of max)", ylabel="throw angle (deg)", title=title)
    for (speed, topspin_factor, cut_angle), (_, _, throw_angles) in results.items():
        label = f"speed={speed:.3} m/s"
        if topspin_factors is not None:
            label += f", topspin_factor={topspin_factor:.2}"
        if cut_angles is not None:
            cut_angle_deg = math.degrees(cut_angle)
            label += f", cut_angle={cut_angle_deg:.3} deg"
        ax.plot(sidespin_percentages, np.rad2deg(throw_angles), label=label)
    ax.legend()
    ax.grid()
    plt.show()


def technical_proof_A14_plots(config: BallBallCollisionExperimentConfig):
    slow_speed = 0.447
    medium_speed = 1.341
    fast_speed = 3.129
    alciatore_speeds = [slow_speed, medium_speed, fast_speed]

    model_str = config.model.model
    friction_str = config.model.friction.model

    title = f"Natural-Roll Shot Collision at Various Speeds\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(title, config, alciatore_speeds, topspin_factors=[1.0])

    title = f"Stun Shot Collision at Various Speeds\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(title, config, alciatore_speeds)

    title = f"Medium-Speed Shot with Various Amounts of Topspin\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(
        title, config, [medium_speed], topspin_factors=np.linspace(0, 1, 5)
    )

    title = f"Medium-Speed Head-On Collision with Various Amounts of Topspin\nThrow Angle vs. Sidespin\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_sidespin_factor(
        title, config, [medium_speed], topspin_factors=np.linspace(0, 1, 5)
    )

    title = f"Medium-Speed Half-Ball Hit with Various Amounts of Topspin\nThrow Angle vs. Sidespin\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_sidespin_factor(
        title,
        config,
        [medium_speed],
        topspin_factors=np.linspace(0, 1, 5),
        cut_angles=[math.radians(30)],
    )

    title = f"Head-On Collision at Various Speeds\nThrow Angle vs. Percent English\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_percent_sidespin(title, config, alciatore_speeds)

    title = f"Half-Ball Hit at Various Speeds\nThrow Angle vs. Percent English\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_percent_sidespin(
        title, config, alciatore_speeds, cut_angles=[math.radians(30)]
    )

    title = f"Slow-Speed Stun Shot with Various Typical Sidespins\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(
        title, config, [slow_speed], sidespin_factors=[0.0, -1.0, 1.0]
    )

    title = f"Slow-Speed Stun Shot with Various 25% Sidespins\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(
        title,
        config,
        [slow_speed],
        sidespin_factors=[
            0.0,
            -cue_strike_spin_rate_factor_percent_english(25),
            cue_strike_spin_rate_factor_percent_english(25),
        ],
    )
    title = f"Medium-Speed Natural-Roll Shot with Various 25% Sidespins\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(
        title,
        config,
        [medium_speed],
        topspin_factors=[1.0],
        sidespin_factors=[
            0.0,
            -cue_strike_spin_rate_factor_percent_english(25),
            cue_strike_spin_rate_factor_percent_english(25),
        ],
    )

    title = f"Slow-Speed Stun Shot with Various 50% Sidespins\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(
        title,
        config,
        [slow_speed],
        sidespin_factors=[
            0.0,
            -cue_strike_spin_rate_factor_percent_english(50),
            cue_strike_spin_rate_factor_percent_english(50),
        ],
    )
    title = f"Medium-Speed Natural-Roll Shot with Various 50% Sidespins\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(
        title,
        config,
        [medium_speed],
        topspin_factors=[1.0],
        sidespin_factors=[
            0.0,
            -cue_strike_spin_rate_factor_percent_english(50),
            cue_strike_spin_rate_factor_percent_english(50),
        ],
    )

    title = f"Slow-Speed Stun Shot with Various 100% Sidespins\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(
        title,
        config,
        [slow_speed],
        sidespin_factors=[
            0.0,
            -cue_strike_spin_rate_factor_percent_english(100),
            cue_strike_spin_rate_factor_percent_english(100),
        ],
    )
    title = f"Medium-Speed Natural-Roll Shot with Various 100% Sidespins\nThrow Angle vs. Cut Angle\n(model={model_str}, model.friction={friction_str})"
    plot_throw_vs_cut_angle(
        title,
        config,
        [medium_speed],
        topspin_factors=[1.0],
        sidespin_factors=[
            0.0,
            -cue_strike_spin_rate_factor_percent_english(100),
            cue_strike_spin_rate_factor_percent_english(100),
        ],
    )


def main():
    ball_params = BallParams.default()
    alciatore_friction = AlciatoreBallBallFriction(a=9.951e-3, b=0.108, c=1.088)
    average_friction = AverageBallBallFriction()
    for model in [
        FrictionalInelastic(friction=alciatore_friction),
        # FrictionalMathavan(friction=alciatore_friction),
        # FrictionalInelastic(friction=average_friction),
        FrictionalMathavan(friction=average_friction),
    ]:
        technical_proof_A14_plots(
            BallBallCollisionExperimentConfig(
                model=model, params=ball_params, xy_line_of_centers_angle_radians=-234
            )
        )


if __name__ == "__main__":
    main()
