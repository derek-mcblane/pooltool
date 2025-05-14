#! /usr/bin/env python

import math

import matplotlib.pyplot as plt
import numpy as np

from pooltool.objects.ball.datatypes import Ball, BallParams
from pooltool.physics.resolve.ball_ball.friction import (
    AlciatoreBallBallFriction,
    AverageBallBallFriction,
)
from pooltool.physics.resolve.ball_ball.frictional_inelastic import FrictionalInelastic
from pooltool.physics.resolve.ball_ball.frictional_mathavan import FrictionalMathavan


def natural_roll(ball_speed: float, ball_radius: float):
    return ball_speed / ball_radius


def spin_from_english_fraction(
    english_fraction: float, ball_speed: float, ball_radius: float
):
    return 1.25 * english_fraction * natural_roll(ball_speed, ball_radius)


def stun_shot(model, ball_params, speeds, xy_line_of_centers_angle=0.0):
    cb = Ball(id="cue", params=ball_params)
    ob = Ball(id="object", params=ball_params)

    cut_angle_in = np.deg2rad(np.linspace(0, 90, 90 * 3, endpoint=False))

    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(
        xlabel="cut_angle (deg)",
        ylabel="throw angle (deg)",
        title=f"Stun-Shot Collision at Various Speeds - Throw Angle vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    ax2.set(
        xlabel="cut_angle (deg)",
        ylabel="sidespin / natural roll",
        title=f"Stun-Shot Collision at Various Speeds - Induced OB Sidespin vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )

    for speed in speeds:
        throw_angle_out = np.empty_like(cut_angle_in)
        induced_sidespin_factor_out = np.empty_like(cut_angle_in)
        for i, cut_angle in enumerate(cut_angle_in):
            ob.state.rvw[0] = np.array([0, 0, cb.params.R])
            cb.set_position_next_to_at_xy_angle(ob, math.pi + xy_line_of_centers_angle)
            cb.setup_motion(speed, xy_line_of_centers_angle + cut_angle, 0.0)

            cb_f, ob_f = model.resolve(cb, ob, inplace=False)
            assert ob_f.state.rvw[1][2] < 1e-9

            throw_angle_out[i] = (
                np.arctan2(ob_f.state.rvw[1][1], ob_f.state.rvw[1][0])
                - xy_line_of_centers_angle
            )
            induced_sidespin_factor_out[i] = ob_f.state.rvw[2][2] / natural_roll(
                speed, cb.params.R
            )

        ax.plot(
            np.rad2deg(cut_angle_in),
            np.rad2deg(throw_angle_out),
            label=f"speed={speed:.3} m/s",
        )
        ax2.plot(
            np.rad2deg(cut_angle_in),
            induced_sidespin_factor_out / (2 * math.pi),
            label=f"speed={speed:.3} m/s",
        )

    ax.legend()
    ax.grid()
    ax2.legend()
    ax2.grid()
    plt.show()


def natural_roll_shot(model, ball_params, speeds, xy_line_of_centers_angle=0.0):
    cb = Ball(id="cue", params=ball_params)
    ob = Ball(id="object", params=ball_params)

    cut_angle_in = np.deg2rad(np.linspace(0, 90, 90 * 3, endpoint=False))

    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(
        xlabel="cut_angle (deg)",
        ylabel="throw angle (deg)",
        title=f"Natural-Roll-Shot Collision at Various Speeds - Throw Angle vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    ax2.set(
        xlabel="cut_angle (deg)",
        ylabel="sidespin / natural roll",
        title=f"Natural-Roll-Shot Collision at Various Speeds - Induced OB Sidespin vs. Cut Angle\n(model={model.model}, model.friction={model.friction})",
    )

    for speed in speeds:
        throw_angle_out = np.empty_like(cut_angle_in)
        induced_sidespin_factor_out = np.empty_like(cut_angle_in)
        for i, cut_angle in enumerate(cut_angle_in):
            ob.state.rvw[0] = np.array([0, 0, cb.params.R])
            cb.set_position_next_to_at_xy_angle(ob, math.pi + xy_line_of_centers_angle)
            cb.setup_motion(
                speed,
                xy_line_of_centers_angle + cut_angle,
                natural_roll(speed, cb.params.R),
            )

            cb_f, ob_f = model.resolve(cb, ob, inplace=False)
            assert ob_f.state.rvw[1][2] < 1e-9

            throw_angle_out[i] = (
                np.arctan2(ob_f.state.rvw[1][1], ob_f.state.rvw[1][0])
                - xy_line_of_centers_angle
            )
            induced_sidespin_factor_out[i] = ob_f.state.rvw[2][2] / natural_roll(
                speed, cb.params.R
            )

        ax.plot(
            np.rad2deg(cut_angle_in),
            np.rad2deg(throw_angle_out),
            label=f"speed={speed:.3} m/s",
        )
        ax2.plot(
            np.rad2deg(cut_angle_in),
            induced_sidespin_factor_out / (2 * math.pi),
            label=f"speed={speed:.3} m/s",
        )

    ax.legend()
    ax.grid()
    ax2.legend()
    ax2.grid()
    plt.show()


def head_on_sidespin_shot(model, ball_params, speeds, xy_line_of_centers_angle=0.0):
    cb = Ball(id="cue", params=ball_params)
    ob = Ball(id="object", params=ball_params)

    cut_angle = math.radians(0)
    english_fraction_in = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    ax.set(
        xlabel="sidespin (fraction of max)",
        ylabel="throw angle (deg)",
        title=f"Head-On Collision at Various Speeds - Throw Angle vs. Sidespin\n(model={model.model}, model.friction={model.friction})",
    )

    for speed in speeds:
        throw_angle_out = np.empty_like(english_fraction_in)
        for i, english_fraction in enumerate(english_fraction_in):
            ob.state.rvw[0] = np.array([0, 0, cb.params.R])
            cb.set_position_next_to_at_xy_angle(ob, math.pi + xy_line_of_centers_angle)
            sidespin = spin_from_english_fraction(english_fraction, speed, cb.params.R)
            cb.setup_motion(speed, xy_line_of_centers_angle + cut_angle, 0.0, sidespin)

            cb_f, ob_f = model.resolve(cb, ob, inplace=False)

            throw_angle_out[i] = (
                np.arctan2(ob_f.state.rvw[1][1], ob_f.state.rvw[1][0])
                - xy_line_of_centers_angle
            )

        ax.plot(
            english_fraction_in,
            np.rad2deg(throw_angle_out),
            label=f"speed={speed:.3} m/s",
        )

    plt.grid()
    plt.legend()
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
        stun_shot(model, ball_params, alciatore_speeds, math.radians(123))
        natural_roll_shot(model, ball_params, alciatore_speeds)
        head_on_sidespin_shot(model, ball_params, alciatore_speeds, math.radians(-123))
