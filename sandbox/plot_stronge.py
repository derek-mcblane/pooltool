import matplotlib.pyplot as plt
import scipy as sp

from pooltool.physics.resolve.stronge_compliant import *

v_n_0 = -3.0
beta_t = 3.5
beta_n = 1.0
mu = 0.2
e_n = 0.85
k_n = 1e7
eta_squared = beta_t / beta_n / 1.1**2
m = 0.170097

beta_t_by_beta_n = beta_t / beta_n
omega_n = frequency_n(beta_n, k_n, m)
omega_t = frequency_t(omega_n, beta_t_by_beta_n, eta_squared)
t_c = compression_duration(omega_n)
t_f = collision_duration(t_c, e_n)

print(f"omega_t / omega_n = {omega_t / omega_n}")


def df_t_per_m(time, t_c, omega_t, beta_t, u_t_2, v_t_2, t_2):
    return (
        omega_t**2
        / beta_t
        * (
            -omega_t * u_t_2 * math.sin(omega_t * (time - t_2))
            - v_t_2 * math.cos(omega_t * (time - t_2))
        )
    )


def df_n_per_m(t_stick, t_c, omega_n, beta_n, e_n, v_n_0):
    return omega_n**2 / beta_n * v_n_0 * C(t_stick, t_c, omega_n, e_n)


def initial_stick():
    v_t_0 = v_n_0 * mu * sp.interpolate.interp1d([0, 1], [0, eta_squared])(0.5)

    t_slip = slip_time_for_initial_stick(
        v_t_0, v_n_0, omega_t, omega_n, mu, e_n, eta_squared, t_c, t_f
    )

    print(f"t_slip / t_c = {t_slip / t_c}")

    ts = np.linspace(0, t_f, 100)
    force_factor = beta_n / (omega_n * -v_n_0)
    f_per_m_ns = np.array(
        [f_per_m_n(t, t_c, v_n_0, beta_n, omega_n, e_n) * force_factor for t in ts]
    )
    f_per_m_t_sticks = np.array(
        [
            f_per_m_t_stick(t, beta_t, omega_t, v_t_0, 0, 0) * force_factor / mu
            for t in ts
        ]
    )

    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(xlabel="non-dimensional time", ylabel="non-dimensional force")
    ax.plot(ts / t_c, f_per_m_ns)
    ax.plot(ts / t_c, -f_per_m_ns, linestyle="dashed")
    ax.plot(ts / t_c, f_per_m_t_sticks)
    ax.grid()
    plt.show()


def initial_slip(interp):
    assert v_n_0 < 0
    v_t_0 = (
        v_n_0
        * mu
        * sp.interpolate.interp1d([0, 1], [eta_squared, (1 + e_n) * beta_t_by_beta_n])(
            interp
        )
    )
    assert v_t_0 < 0

    ts = np.linspace(0, t_f, 500)
    force_factor = beta_n / (omega_n * -v_n_0)

    t_stick = stick_time_for_initial_slip(
        v_t_0, v_n_0, omega_n, beta_t_by_beta_n, mu, e_n, eta_squared
    )
    print(f"t_stick / t_c = {t_stick / t_c}")

    u_t_slip_to_stick = u_t_initial_slip(
        t_stick, t_c, v_n_0, mu, eta_squared, omega_n, e_n
    )
    v_t_slip_to_stick = v_t_initial_slip(
        t_stick, t_c, v_t_0, v_n_0, mu, beta_t_by_beta_n, omega_n, e_n
    )

    f_per_m_ns = np.array(
        [f_per_m_n(t, t_c, v_n_0, beta_n, omega_n, e_n) * force_factor for t in ts]
    )
    f_per_m_t_sticks = np.array(
        [
            f_per_m_t_stick(
                t, beta_t, omega_t, v_t_slip_to_stick, u_t_slip_to_stick, t_stick
            )
            * force_factor
            / mu
            for t in ts
        ]
    )

    t_slip = slip_time_for_stick(
        v_n_0,
        omega_t,
        omega_n,
        mu,
        e_n,
        eta_squared,
        u_t_slip_to_stick,
        v_t_slip_to_stick,
        t_stick,
        t_c,
        t_f,
    )
    print(f"t_slip / t_c = {t_slip / t_c}")

    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(xlabel="non-dimensional time", ylabel="non-dimensional force")
    ax.plot(ts / t_c, f_per_m_ns)
    ax.plot(ts / t_c, -f_per_m_ns, linestyle="dashed")
    ax.plot(ts / t_c, f_per_m_t_sticks)
    ax.grid()
    plt.show()


if __name__ == "__main__":
    # initial_stick()
    initial_slip(0.01)
    initial_slip(0.25)
    initial_slip(0.5)
    initial_slip(0.75)
    initial_slip(0.99)
