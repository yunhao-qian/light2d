"""
Definition of the constant light material.
"""

from typing import Callable

import numpy as np
from numba import njit

from ..core.base import Material, Ray, SurfaceInteraction


class ConstantLight(Material):
    """
    Constant light is an emissive material with a constant light intensity. On a ray-entity
    intersection, a fixed light intensity specified by the user is returned and the ray is never
    scattered.
    """

    def __init__(self, li: tuple[float, float, float]):
        """
        Creates a constant light whose light intensity is specified by the argument `li`.
        """
        super().__init__()
        self._li = np.array(li, np.float32)

    def _make_scatter_function(self) -> Callable[[Ray, SurfaceInteraction], None]:
        li = self._li

        @njit
        def scatter(ray: Ray, interaction: SurfaceInteraction) -> None:
            interaction[4:7] = li
            interaction[7:10] = -np.inf

        return scatter
