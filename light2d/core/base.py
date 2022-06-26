"""
Type definitions and base classes.
"""

from abc import ABC, abstractmethod
from typing import Callable, NewType

import numpy as np
import numpy.typing as npt


F32Array = npt.NDArray[np.float32]

AlignedBox = NewType('AlignedBox', F32Array)
"""
An axis-aligned box is represented by a 2x2 float32 array.

* The first row of this array is the minimum x and y coordinates.
* The second row of this array is the maximum x and y coordinates.
"""

Ray = NewType('Ray', F32Array)
"""
A ray is represented by a 1-D float32 array of length 5.

* `o`: Origin of the ray. It takes elements [0, 2) of this array.
* `d`: Direction of the ray which may not necessarily be normalized. It takes elements [2, 4) of
  this array.
* `t_max`: Maximum distance from the origin where an intersection may occur. It takes element 5 of
  this array. Note that distances in a ray is measured with respect to the direction vector's
  length. This means that the farthest point where an intersection may occur is `o + d * t_max`.
"""

SurfaceInteraction = NewType('SurfaceInteraction', F32Array)
"""
A surface interaction is represented by a 1-D float32 array of length 12.

* `p`: Point of the ray-entity intersection. It takes elements [0, 2) of this array.
* `n`: Outward-pointing normal vector of the surface which may not necessarily be normalized. It
  takes elements [2, 4) of this array.
* `li`: Light intensity from the material. It takes elements [4, 7) of this array. It should be set
  to zeros if the material is not emissive.
* `attenuation`: Attenuation applied to the traced light intensity of the scattered ray. It takes
  elements [7, 10) of this array. If none of its components is positive, the ray will not be
  scattered and the `d_out` field is undefined.
* `d_out`: If the ray will be scattered, this fields represents the direction of the scattered ray.
  It takes elements [10, 12) of this array.
"""


class Shape(ABC):
    """
    Base class of shapes. A shape has an axis-aligned bounding box and can be intersected by a ray.
    """

    @property
    @abstractmethod
    def bounds(self) -> AlignedBox:
        """
        Returns the axis-aligned bounding box of this shape.
        """
        ...

    @abstractmethod
    def intersect_function(self) -> Callable[[Ray, SurfaceInteraction], bool]:
        """
        Returns a JIT-ed function for intersection tests of this shape.

        * The first argument of the returned function is a ray. Its `t_max` field will be updated
          on an intersection.
        * The second argument of the returned function is a surface interaction. Its `p` and `n`
          fields will be updated on an intersection. Remaining fields of the surface interaction are
          left unchanged and should be updated later by a material instance.
        * The return value of the returned function indicates whether there is an intersection.
        """
        ...


class Material(ABC):
    """
    Base class of materials. A material can be emissive, and may scatter a ray on a ray-entity
    intersection.
    """

    @abstractmethod
    def scatter_function(self) -> Callable[[Ray, SurfaceInteraction], None]:
        """
        Returns a JIT-ed function for calculations related to this material.

        * The first argument of the returned function is a ray. Its value should not be changed by
          this function.
        * The second argument of the returned function is a surface interaction. Its `p` and `n`
          fields (which should already been set by a shape instance) are not changed, while its
          `li`, `attenuation`, and `d_out` fields are updated.
        """
        ...


class Entity(ABC):
    """
    Base class of entities. An entity is the combination of consisting shapes and materials.
    """

    @abstractmethod
    def intersect_function(self) -> Callable[[Ray, SurfaceInteraction], bool]:
        """
        Returns a JIT-ed function for intersection tests and material calculations related to this
        entity.

        * The first argument of the returned function is a ray. Its `t_max` fields will be updated
          on an intersection.
        * The second argument of the returned function is a surface interaction. All of its fields
          will be updated on an intersection.
        * The return value of the returned function indicates whether there is an intersection.
        """
        ...


class Integrator(ABC):
    """
    Base class of integrators. An integrator calculates the light intensity of each pixel, which
    directly corresponds to the pixel color in the output picture.
    """

    @abstractmethod
    def integrate_function(self, entity: Entity) -> Callable[[AlignedBox], F32Array]:
        """
        Given an entity, returns a JIT-ed function for calculating the light intensity of a given
        pixel.

        * The first (and only) argument of the returned function is an axis-aligned box representing
          the region occupied by this pixel.
        * The return value of the returned function is a 1-D float32 array of length 3 representing
          the light intensity of this pixel.
        """
        ...
