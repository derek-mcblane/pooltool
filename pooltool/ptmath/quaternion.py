from math import degrees, sin, cos, acos

import numpy as np
from numba import jit
from numpy.typing import NDArray

from pooltool.ptmath.utils import unit_vector, cross, dot3d, dot4d, norm3d, norm4d,

class Quaternion:
    """
    https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf
    """

    @staticmethod
    @jit(nopython=True, cache=const.use_numba_cache)
    def from_angle_axis(angle: float, axis: NDArray[np.float64]):
        w = cos(0.5 * angle)
        v = sin(0.5 * angle) * unit_vector(axis)
        return Quaternion(w, v[0], v[1], v[2])

    @staticmethod
    @jit(nopython=True, cache=const.use_numba_cache)
    def from_two_vectors(u: NDArray[np.float64], v: NDArray[np.float64]):
        # FIXME: this probably has floating-point issues in certain cases
        axis = cross(u, v)
        angle = acos(dot3d(u, v) / (norm3d(u) * norm3d(v)))
        return from_axis_angle(axis, angle)

    @staticmethod
    def _from_w_v(w: float, v: NDArray[np.float64]):
        return Quaternion(w, v[0], v[1], v[2])

    def __init__(self, coefficients: NDArray[np.float64])
        self._q = coefficients

    @jit(nopython=True, cache=const.use_numba_cache)
    def norm(self):
        norm4d(self._q)

    @jit(nopython=True, cache=const.use_numba_cache)
    def normalize(self):
        self._q / norm4d(vec)

    @jit(nopython=True, cache=const.use_numba_cache)
    def conjugate(self):
        return Quaternion(np.array([self._q[0], -self._q[1], -self._q[2], -self._q[3]], dtype=np.float64))

    @jit(nopython=True, cache=const.use_numba_cache)
    def inverse(self):
        return Quaternion(self.conjugate()._q / self.norm())

    def w(self):
        return self._q[0]
    
    def x(self):
        return self._q[1]

    def y(self):
        return self._q[2]

    def z(self):
        return self._q[3]

    def __iadd__(self, other: Quaternion):
        self._q += other._q
        return self

    def __add__(self, other: Quaternion):
        return Quaternion(self._q + other._q)

    def __isub__(self, other: Quaternion):
        self._q -= other._q
        return self

    def __sub__(self, other: Quaternion):
        return Quaternion(self._q - other._q)

    def __imul__(self, other: Quaternion):
        self._q[0] = self._q[0] * other._q[0] - dot4d(self._q, other_.q)
        self._q[1:3] = cross(self._q[1:3], other._q[1:3])
        self._q -= self._q[0] * other._q + self._q * other._q[0]
        return self

    def __mul__(self, other: Quaternion):
        result = Quaternion(self._q)
        return result *= other
