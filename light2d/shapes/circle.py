"""
Definition of the circle shape.
"""

from typing import Callable

import numpy as np
from numba import njit

from ..core.base import AlignedBox, Ray, Shape, SurfaceInteraction


class Circle(Shape):
    """
    Circle is a shape specified by its center and radius.
    """

    def __init__(self, center: tuple[float, float], radius: float):
        """
        Creates a circle with the specified center and radius.
        """
        super().__init__()
        self._center = np.array(center, np.float32)
        self._radius = np.float32(radius)

    def _get_bounds(self) -> AlignedBox:
        return np.stack((self._center - self._radius, self._center + self._radius))

    def _make_intersect_function(self) -> Callable[[Ray, SurfaceInteraction], bool]:
        center = self._center
        radius = self._radius

        @njit
        def intersect(ray: Ray, interaction: SurfaceInteraction) -> bool:
            o = ray[0:2]
            d = ray[2:4]
            d_norm = np.linalg.norm(d)
            d_normalized = d / d_norm
            t_max = ray[4]

            oc = center - o
            d_oc = np.dot(d_normalized, oc)
            delta = np.square(d_oc) - np.dot(oc, oc) + np.square(radius)
            if not delta >= 0:
                return False

            sqrt_delta = np.sqrt(delta)
            t1 = (d_oc - sqrt_delta) / d_norm
            if not t1 < t_max:
                return False
            if 0 < t1:
                t = t1
            else:
                t2 = (d_oc + sqrt_delta) / d_norm
                if 0 < t2 < t_max:
                    t = t2
                else:
                    return False

            p = o + d * t
            n = p - center
            n /= np.linalg.norm(n)
            ray[4] = t
            interaction[0:2] = p
            interaction[2:4] = n
            return True

        return intersect
