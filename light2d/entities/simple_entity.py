"""
Definition of the simple entity.
"""

from typing import Callable

from numba import njit

from ..core.base import AlignedBox, Entity, Material, Ray, Shape, SurfaceInteraction


class SimpleEntity(Entity):
    """
    Simple entity is an entity composed of a single shape and a single material.
    """

    def __init__(self, shape: Shape, material: Material):
        """
        Creates a simple entity with the specified shape and material.
        """
        super().__init__()
        self._shape = shape
        self._material = material

    def _get_bounds(self) -> AlignedBox:
        return self._shape.bounds

    def _make_intersect_function(self) -> Callable[[Ray, SurfaceInteraction], bool]:
        shape_intersect = self._shape.intersect_function
        material_scatter = self._material.scatter_function

        @njit
        def intersect(ray: Ray, interaction: SurfaceInteraction) -> bool:
            if not shape_intersect(ray, interaction):
                return False
            material_scatter(ray, interaction)
            return True

        return intersect
