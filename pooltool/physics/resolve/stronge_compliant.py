import logging
import math
from typing import Tuple

import scipy as sp
from numba import jit

import pooltool.constants as const

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def find_zero(f, a, b, xtol, maxiter=1000):
    assert a <= b, f"a={a} b={b}"
    x = None
    for _ in range(10):
        if (a - b) - xtol <= 0:
            return (a + b) / 2
        try:
            x = sp.optimize.toms748(f, a, b, xtol=xtol, maxiter=maxiter)
            break
        except ValueError:
            steps = 1000
            a, b = math.nextafter(a, b, steps=steps), math.nextafter(b, a, steps=steps)
    if x is None:
        x = sp.optimize.toms748(f, a, b, xtol=xtol, maxiter=maxiter)
    return x


@jit(nopython=True, cache=const.use_numba_cache)
def normal_tangent_stiffness_ratio(poisson_ratio):
    return (2 - poisson_ratio) / (2 * (1 - poisson_ratio))


@jit(nopython=True, cache=const.use_numba_cache)
def t_c_shift(e_n):
    return (math.pi / 2) * (1 - 1 / e_n)


@jit(nopython=True, cache=const.use_numba_cache)
def C_compression(time, omega_n):
    return math.cos(omega_n * time)


@jit(nopython=True, cache=const.use_numba_cache)
def C_restitution(time, omega_n, e_n):
    return e_n * math.cos(time * omega_n / e_n + t_c_shift(e_n))


@jit(nopython=True, cache=const.use_numba_cache)
def C(time, t_c, omega_n, e_n):
    if time <= t_c:
        return C_compression(time, omega_n)
    else:
        return C_restitution(time, omega_n, e_n)


@jit(nopython=True, cache=const.use_numba_cache)
def v_n_compression(time, v_n_0, omega_n):
    return v_n_0 * math.cos(omega_n * time)


@jit(nopython=True, cache=const.use_numba_cache)
def v_n_restitution(time, v_n_0, omega_n, e_n):
    return e_n * v_n_0 * math.cos(omega_n * time / e_n + t_c_shift(e_n))


@jit(nopython=True, cache=const.use_numba_cache)
def v_n(time, compression_duration, v_n_0, omega_n, e_n):
    if time <= compression_duration:
        return v_n_compression(time, v_n_0, omega_n)
    else:
        return v_n_restitution(time, v_n_0, omega_n, e_n)


@jit(nopython=True, cache=const.use_numba_cache)
def f_per_m_n_compression(time, v_n_0, beta_n, omega_n):
    assert time >= 0
    assert v_n_0 < 0
    assert beta_n > 0
    assert omega_n > 0
    return -omega_n * v_n_0 / beta_n * math.sin(omega_n * time)


@jit(nopython=True, cache=const.use_numba_cache)
def f_per_m_n_restitution(time, v_n_0, beta_n, omega_n, e_n):
    assert time >= 0
    assert v_n_0 < 0
    assert beta_n > 0
    assert omega_n > 0
    assert e_n > 0
    return -omega_n * v_n_0 / beta_n * math.sin(omega_n * time / e_n + t_c_shift(e_n))


@jit(nopython=True, cache=const.use_numba_cache)
def f_per_m_n(time, compression_duration, v_n_0, beta_n, omega_n, e_n):
    if time <= compression_duration:
        return f_per_m_n_compression(time, v_n_0, beta_n, omega_n)
    else:
        return f_per_m_n_restitution(time, v_n_0, beta_n, omega_n, e_n)


@jit(nopython=True, cache=const.use_numba_cache)
def p_per_m_n_compression(time, v_n_0, beta_n, omega_n):
    return -v_n_0 / beta_n * (1 - math.cos(omega_n * time))


@jit(nopython=True, cache=const.use_numba_cache)
def p_per_m_n_restitution(time, v_n_0, beta_n, omega_n, e_n):
    return -v_n_0 / beta_n * (1 - e_n * math.cos(omega_n * time / e_n + t_c_shift(e_n)))


@jit(nopython=True, cache=const.use_numba_cache)
def p_per_m_n(time, t_c, v_n_0, beta_n, omega_n, e_n):
    if time <= t_c:
        return p_per_m_n_compression(time, v_n_0, beta_n, omega_n)
    else:
        return p_per_m_n_restitution(time, v_n_0, beta_n, omega_n, e_n)


@jit(nopython=True, cache=const.use_numba_cache)
def u_t_initial_slip_compression(time, v_n_0, mu, eta_squared, omega_n):
    return mu * v_n_0 * eta_squared / omega_n * math.sin(omega_n * time)


@jit(nopython=True, cache=const.use_numba_cache)
def u_t_initial_slip_restitution(time, v_n_0, mu, eta_squared, omega_n, e_n):
    return (
        mu
        * v_n_0
        * eta_squared
        / omega_n
        * math.sin(omega_n * time / e_n + t_c_shift(e_n))
    )


@jit(nopython=True, cache=const.use_numba_cache)
def u_t_initial_slip(time, t_c, v_n_0, mu, eta_squared, omega_n, e_n):
    if time <= t_c:
        return u_t_initial_slip_compression(time, v_n_0, mu, eta_squared, omega_n)
    else:
        return u_t_initial_slip_restitution(time, v_n_0, mu, eta_squared, omega_n, e_n)


@jit(nopython=True, cache=const.use_numba_cache)
def v_t_initial_slip_compression(time, v_t_0, v_n_0, mu, beta_t_by_beta_n, omega_n):
    return v_t_0 - mu * beta_t_by_beta_n * v_n_0 * (1 - math.cos(omega_n * time))


@jit(nopython=True, cache=const.use_numba_cache)
def v_t_initial_slip_restitution(
    time, v_t_0, v_n_0, mu, beta_t_by_beta_n, omega_n, e_n
):
    return v_t_0 - mu * beta_t_by_beta_n * v_n_0 * (
        1 - e_n * math.cos(omega_n * time / e_n + t_c_shift(e_n))
    )


@jit(nopython=True, cache=const.use_numba_cache)
def v_t_initial_slip(time, t_c, v_t_0, v_n_0, mu, beta_t_by_beta_n, omega_n, e_n):
    if time <= t_c:
        return v_t_initial_slip_compression(
            time, v_t_0, v_n_0, mu, beta_t_by_beta_n, omega_n
        )
    else:
        return v_t_initial_slip_restitution(
            time, v_t_0, v_n_0, mu, beta_t_by_beta_n, omega_n, e_n
        )


@jit(nopython=True, cache=const.use_numba_cache)
def u_t_stick(time, omega_t, v_t_stick_0, u_t_stick_0=0.0, time_stick=0.0):
    return u_t_stick_0 * math.cos(
        omega_t * (time - time_stick)
    ) - v_t_stick_0 / omega_t * math.sin(omega_t * (time - time_stick))


@jit(nopython=True, cache=const.use_numba_cache)
def v_t_stick(time, omega_t, v_t_stick_0, u_t_stick_0=0.0, time_stick=0.0):
    return omega_t * u_t_stick_0 * math.sin(
        omega_t * (time - time_stick)
    ) + v_t_stick_0 * math.cos(omega_t * (time - time_stick))


@jit(nopython=True, cache=const.use_numba_cache)
def f_per_m_t_stick(time, beta_t, omega_t, v_t_stick, u_t_stick, time_stick):
    return (
        omega_t**2 * u_t_stick * math.cos(omega_t * (time - time_stick))
        - omega_t * v_t_stick * math.sin(omega_t * (time - time_stick))
    ) / beta_t


@jit(nopython=True, cache=const.use_numba_cache)
def f_per_m_t_slip(time, compression_duration, v_n_0, beta_n, omega_n, mu, e_n):
    return mu * f_per_m_n(time, compression_duration, v_n_0, beta_n, omega_n, e_n)


@jit(nopython=True, cache=const.use_numba_cache)
def is_initial_stick(v_t_0, v_n_0, mu, eta_squared):
    return v_t_0 / v_n_0 < mu * eta_squared


@jit(nopython=True, cache=const.use_numba_cache)
def is_gross_slip(v_t_0, v_n_0, mu, e_n, beta_t_by_beta_n):
    return v_t_0 / v_n_0 > mu * (1 + e_n) * beta_t_by_beta_n


def slip_time_for_initial_stick(
    v_t_0: float,
    v_n_0: float,
    omega_t: float,
    omega_n: float,
    mu: float,
    e_n: float,
    eta_squared: float,
    t_c: float,
    t_f: float,
) -> float:
    def f(t):
        return abs(-v_t_0 * math.sin(omega_t * t) / v_n_0) - (
            mu
            * eta_squared
            * omega_t
            / omega_n
            * math.sin(omega_n * t / e_n + t_c_shift(e_n))
        )

    def f_t_nondim(t_nondim):
        return f(t_nondim * t_c)

    t_slip = t_c * sp.optimize.newton(f_t_nondim, 1, x1=1.01)
    assert t_c <= t_slip and t_slip <= t_f, f"t_c={t_c} <= t_slip={t_slip} <= t_f={t_f}"
    return t_slip


def slip_time_for_stick(
    v_n_0: float,
    omega_t: float,
    omega_n: float,
    mu: float,
    e_n: float,
    eta_squared: float,
    u_t_stick: float,
    v_t_stick: float,
    t_stick: float,
    t_c: float,
    t_f: float,
) -> float:
    def f(t):
        return abs(
            omega_n
            / (mu * v_n_0)
            * (
                u_t_stick * math.cos(omega_t * (t - t_stick))
                - v_t_stick / omega_t * math.sin(omega_t * (t - t_stick))
            )
        ) - eta_squared * math.sin(omega_n * t / e_n + t_c_shift(e_n))

    def f_t_nondim(t_nondim):
        return f(t_nondim * t_c)

    bracket_min = t_stick / t_c
    bracket_max = t_f / t_c
    result = t_c * sp.optimize.root_scalar(
        f_t_nondim, bracket=[bracket_min, bracket_max]
    )
    t_slip = result.root

    assert t_stick <= t_slip and t_slip <= t_f, (
        f"t_stick={t_stick} <= t_slip={t_slip} <= t_f={t_f}"
    )
    return t_slip


@jit(nopython=True, cache=const.use_numba_cache)
def stick_time_for_initial_slip(
    v_t_0: float,
    v_n_0: float,
    omega_n: float,
    beta_t_by_beta_n: float,
    mu: float,
    e_n: float,
    eta_squared: float,
) -> float:
    t_c = compression_duration(omega_n)
    t_f = collision_duration(t_c, e_n)
    v_t_by_v_n = v_t_0 / v_n_0
    assert mu * eta_squared <= v_t_by_v_n
    assert 0 <= v_t_by_v_n and v_t_by_v_n <= mu * (1 + e_n) * beta_t_by_beta_n
    if v_t_by_v_n <= mu * beta_t_by_beta_n:
        x = (v_t_by_v_n / mu - beta_t_by_beta_n) / (eta_squared - beta_t_by_beta_n)
        assert 0 <= x and x <= 1, (
            f"x = {x}, (mu * eta_squared = {mu * eta_squared}) <= (v_t_0_by_v_n_0 = {v_t_by_v_n}) <= (mu * (1 + e_n) * beta_t_by_beta_n = {mu * (1 + e_n) * beta_t_by_beta_n})"
        )
        result = math.acos(x) / omega_n
        assert 0 <= result and result <= t_c
    else:
        x = (v_t_by_v_n / mu - beta_t_by_beta_n) / (
            eta_squared / e_n - beta_t_by_beta_n * e_n
        )
        assert -1 <= x and x <= 0, (
            f"x = {x}, (mu * eta_squared = {mu * eta_squared}) <= (v_t_0_by_v_n_0 = {v_t_by_v_n}) <= (mu * (1 + e_n) * beta_t_by_beta_n = {mu * (1 + e_n) * beta_t_by_beta_n})"
        )
        result = (math.acos(x) - t_c_shift(e_n)) * e_n / omega_n
        assert t_c <= result and result <= t_f
    return result


@jit(nopython=True, cache=const.use_numba_cache)
def frequency_t(omega_n, beta_t_by_beta_n, eta_squared):
    return omega_n * math.sqrt(beta_t_by_beta_n / eta_squared)


@jit(nopython=True, cache=const.use_numba_cache)
def frequency_n(beta_n, k_n, m):
    return math.sqrt(beta_n * k_n / m)


@jit(nopython=True, cache=const.use_numba_cache)
def compression_duration(omega_n):
    return math.pi / (2 * omega_n)


@jit(nopython=True, cache=const.use_numba_cache)
def collision_duration(t_c, e_n):
    return (1 + e_n) * t_c


def resolve_collinear_compliant_frictional_inelastic_collision(
    v_t_0: float,
    v_n_0: float,
    m: float,
    beta_t: float,
    beta_n: float,
    mu: float,
    e_n: float,
    k_n: float,
    eta_squared: float,
) -> Tuple[float, float]:
    assert v_t_0 <= 0
    assert v_n_0 <= 0

    beta_t_by_beta_n = beta_t / beta_n
    omega_n = frequency_n(beta_n, k_n, m)
    omega_t = frequency_t(omega_n, beta_t_by_beta_n, eta_squared)
    assert omega_t / omega_n > 1
    assert omega_t / omega_n < 2
    t_c = compression_duration(omega_n)
    t_f = collision_duration(t_c, e_n)

    if is_gross_slip(v_t_0, v_n_0, mu, e_n, beta_t_by_beta_n):
        logger.debug("gross slip case")
        v_t_f = v_t_0 - mu * v_n_0 * beta_t_by_beta_n * (1 + e_n)
    elif is_initial_stick(v_t_0, v_n_0, mu, eta_squared):
        logger.debug("stick-slip case")
        t_slip = slip_time_for_initial_stick(
            v_t_0, v_n_0, omega_t, omega_n, mu, e_n, eta_squared, t_c, t_f
        )
        v_t_stick_to_slip = v_t_stick(t_slip, omega_t, v_t_0)
        # error in book? has B_1/B_2 instead of B_1/B_3
        v_t_f = v_t_stick_to_slip + mu * v_n_0 * beta_t_by_beta_n * e_n * (
            1 + math.cos(omega_n * t_slip / e_n + t_c_shift(e_n))
        )
    else:
        logger.debug("slip-stick-slip case")
        t_stick = stick_time_for_initial_slip(
            v_t_0, v_n_0, omega_n, beta_t_by_beta_n, mu, e_n, eta_squared
        )
        u_t_slip_to_stick = u_t_initial_slip(
            t_stick, t_c, v_n_0, mu, eta_squared, omega_n, e_n
        )
        v_t_slip_to_stick = v_t_initial_slip(
            t_stick, t_c, v_t_0, v_n_0, mu, beta_t_by_beta_n, omega_n, e_n
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
        v_t_stick_to_slip = v_t_stick(
            t_slip, omega_t, v_t_slip_to_stick, u_t_slip_to_stick, t_stick
        )
        p_per_m_n_slip = p_per_m_n_restitution(t_slip, v_n_0, beta_n, omega_n, e_n)
        p_per_m_n_c = p_per_m_n_compression(t_c, v_n_0, beta_n, omega_n)
        v_t_f = v_t_stick_to_slip + mu * v_n_0 * beta_t_by_beta_n * (
            1 + e_n - p_per_m_n_slip / p_per_m_n_c
        )
        """
        p_per_m_n_f = p_per_m_n_restitution(t_f, v_n_0, beta_n, omega_n, e_n)
        v_t_f = v_t_stick_to_slip - mu * beta_t * (p_per_m_n_f - p_per_m_n_slip)
        """

    v_n_f = v_n_restitution(t_f, v_n_0, omega_n, e_n)

    logger.debug(
        f"v_t_0/(mu * v_n_0)={v_t_0 / (mu * v_n_0)}, v_t_f/(mu * v_n_0)={v_t_f / (mu * v_n_0)}"
    )
    if v_t_f > 0:
        logger.debug("v_t reversed direction!")

    return v_t_f, v_n_f
