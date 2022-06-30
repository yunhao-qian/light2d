"""
Definition of the flat aggregate.
"""

from typing import Callable, Iterable

from numba import literal_unroll, njit

from ..core.base import AlignedBox, Entity, Ray, SurfaceInteraction
from ..core.utils import aligned_box_union


class FlatAggregate(Entity):
    """
    Flat aggregate is a flat collection of entities without an acceleration structure.
    """

    def __init__(self, entities: Iterable[Entity]):
        """
        Creates a flat aggregate from the given iterable of consisting entities.
        """
        super().__init__()
        self._entities = list(entities)

    def _get_bounds(self) -> AlignedBox:
        return aligned_box_union(e.bounds for e in self._entities)

    def _make_intersect_function(self) -> Callable[[Ray, SurfaceInteraction], bool]:
        entities_intersect = tuple(e.intersect_function for e in self._entities)

        @njit
        def intersect(ray: Ray, interaction: SurfaceInteraction) -> bool:
            is_intersected = False
            for entity_intersect in literal_unroll(entities_intersect):
                is_intersected = is_intersected or entity_intersect(ray, interaction)
            return is_intersected

        return intersect
