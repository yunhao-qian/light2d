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
        self.shape = shape
        self.material = material

    @property
    def bounds(self) -> AlignedBox:
        return self.shape.bounds

    def intersect_function(self) -> Callable[[Ray, SurfaceInteraction], bool]:
        shape_intersect = self.shape.intersect_function()
        material_scatter = self.material.scatter_function()

        @njit
        def intersect(ray: Ray, interaction: SurfaceInteraction) -> bool:
            if not shape_intersect(ray, interaction):
                return False
            material_scatter(ray, interaction)
            return True

        return intersect
