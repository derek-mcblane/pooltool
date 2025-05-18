import matplotlib.pyplot as plt
import numpy as np

import pooltool as pt

template = pt.System.example()
template.balls["cue"].state.rvw[0, 0] = 0.3
template.cue.set_state(a=0.4, b=0.126, V0=3.3, theta=15.0)

systems = []
systems_n_ball_collisions = []

n_phi = 100
phi_in = np.linspace(5.5, 6.8, n_phi)
cut_angle_in = np.empty(n_phi)
throw_angle_out = np.empty(n_phi)
ob_corkscrew_out = np.empty(n_phi)
ob_topspin_out = np.empty(n_phi)
ob_sidespin_out = np.empty(n_phi)

fig = plt.figure(1)
ax = fig.add_subplot(1, 4, 1)
ax.set(
    xlabel="cut_angle (deg)",
    ylabel="throw angle (deg)",
    title=f"OB Throw Angle vs. Cut Angle",
)
ax2 = fig.add_subplot(1, 4, 2)
ax2.set(
    xlabel="cut_angle (deg)",
    ylabel="corkscrew spin (rev/s)",
    title=f"OB Corkscrew vs. Cut Angle",
)
ax3 = fig.add_subplot(1, 4, 3)
ax3.set(
    xlabel="cut_angle (deg)",
    ylabel="top spin (rev/s)",
    title=f"OB Topspin vs. Cut Angle",
)
ax4 = fig.add_subplot(1, 4, 4)
ax4.set(
    xlabel="cut_angle (deg)",
    ylabel="side spin (rev/s)",
    title=f"OB Sidespin vs. Cut Angle",
)

for i, phi_delta in enumerate(phi_in):
    system = template.copy()

    system.cue.set_state(phi=(pt.aim.at_ball(system, "1") + phi_delta))

    pt.simulate(system, inplace=True, continuous=True, dt=0.01)
    ball_ball_collision = pt.events.filter_type(system.events, pt.EventType.BALL_BALL)[0]

    cb_i = ball_ball_collision.get_ball("cue", True)
    ob_i = ball_ball_collision.get_ball("1", True)

    cb_v_i_angle = np.arctan2(cb_i.vel[1], cb_i.vel[0])
    loc = ob_i.xyz - cb_i.xyz
    loc_angle = np.arctan2(loc[1], loc[0])
    cut_angle_in[i] = cb_v_i_angle - loc_angle

    cb_f = ball_ball_collision.get_ball("cue", False)
    ob_f = ball_ball_collision.get_ball("1", False)

    ob_v_f_angle = np.arctan2(ob_f.vel[1], ob_f.vel[0])
    throw_angle_out[i] = ob_v_f_angle - loc_angle
    ob_w = pt.ptmath.coordinate_rotation(ob_f.avel, ob_v_f_angle)
    ob_corkscrew_out[i] = ob_w[0]
    ob_topspin_out[i] = ob_w[1]
    ob_sidespin_out[i] = ob_w[2]

ax.plot(np.rad2deg(cut_angle_in), np.rad2deg(throw_angle_out))
ax2.plot(np.rad2deg(cut_angle_in), ob_corkscrew_out)
ax3.plot(np.rad2deg(cut_angle_in), ob_topspin_out)
ax4.plot(np.rad2deg(cut_angle_in), ob_sidespin_out)

ax.grid()
ax2.grid()
ax3.grid()
ax4.grid()

plt.show()
