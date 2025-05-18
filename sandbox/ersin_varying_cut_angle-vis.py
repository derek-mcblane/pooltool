import numpy as np

import pooltool as pt

template = pt.System.example()
template.balls["cue"].state.rvw[0, 0] = 0.3
template.cue.set_state(a=0.4, b=0.126, V0=3.3, theta=15.0)

systems = []
systems_n_ball_collisions = []

n_phi = 100
phi_in = np.linspace(5.8, 6.8, n_phi)
for i, phi_delta in enumerate(phi_in):
    system = template.copy()

    system.cue.set_state(phi=(pt.aim.at_ball(system, "1") + phi_delta))

    pt.simulate(system, inplace=True, continuous=True, dt=0.01)
    systems.append(system)

pt.show(pt.MultiSystem(systems))
