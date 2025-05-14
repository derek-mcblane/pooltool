#! /usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np

from pooltool.objects.ball.datatypes import Ball
from pooltool.objects.cue.datatypes import Cue
from pooltool.physics.resolve.stick_ball.instantaneous_point import InstantaneousPoint


def tip_offset_with_varying_elevation(cue_a, cue_b, axes):
    """Simulate cuestick-ball collisions with a constant contact-point offset in the cue frame and varying cue elevation."""

    cue = Cue(V0=5.0, a=cue_a, b=cue_b)
    ball = Ball(id="ball")
    resolver = InstantaneousPoint()

    thetas = np.linspace(0, 90, 90 + 1)
    angular_velocities = np.zeros([thetas.size, 3])

    for i, theta in enumerate(thetas):
        cue.set_state(theta=theta)
        _, ball_f = resolver.resolve(cue, ball)
        angular_velocities[i, :] = ball_f.avel

    axes.set(
        xlabel="theta (degrees)",
        ylabel="angular velocity (rad / s)",
        title=f"Elevation vs. Spin Rate, cue_a={cue_a} cue_b={cue_b}",
    )

    axes.plot(
        thetas,
        angular_velocities[:, 0],
        label="corkscrew spin (ball-frame x angular velocity)",
    )
    axes.plot(
        thetas,
        angular_velocities[:, 1],
        label="top spin(ball-frame y angular velocity)",
    )
    axes.plot(
        thetas,
        angular_velocities[:, 2],
        label="side spin (ball-frame z angular velocity)",
    )


if __name__ == "__main__":
    # center ball hit
    plt.figure()
    ax = plt.subplot()
    tip_offset_with_varying_elevation(0, 0, ax)
    plt.legend()
    # horizontal contact-point offset only
    plt.figure()
    ax = plt.subplot()
    tip_offset_with_varying_elevation(-0.5, 0, ax)
    plt.legend()
    # vertical contact-point offset only
    plt.figure()
    ax = plt.subplot()
    tip_offset_with_varying_elevation(0, 0.5, ax)
    plt.legend()
    # combined contact-point offset
    plt.figure()
    ax = plt.subplot()
    tip_offset_with_varying_elevation(-0.25, -0.25, ax)
    plt.legend()

    plt.show()
