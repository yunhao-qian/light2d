"""
Type definitions and base classes.
"""

from abc import ABC, abstractmethod
from typing import Callable, NewType

import numpy as np
import numpy.typing as npt


EPSILON = np.float32(1e-4)
"""
A parameter used to eliminate self-intersections of scattered rays due to rounding errors of
floating-point numbers.

For every scattered ray, the origin is shifted from the intersection point along the normal
direction of the surface (or its opposite, depending on the direction of the scattered ray) by a
distance of `EPSILON`.
"""

F32Array = npt.NDArray[np.float32]

Spectrum = NewType('Spectrum', F32Array)
"""
A spectrum is represented by a 1-D float32 array of length 3.

* `r`: Red channel. It takes element 0 of this array.
* `g`: Green channel. It takes element 1 of this array.
* `b`: Blue channel. It takes element 2 of this array.
"""

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

    def __init__(self):
        self._bounds = None
        self._intersect_function = None

    @abstractmethod
    def _get_bounds(self) -> AlignedBox:
        """
        Calculates the axis-aligned bounding box of this shape.

        Users should not call this method directly. Use the `bounds` property instead.
        """
        ...

    @property
    def bounds(self) -> AlignedBox:
        """
        Returns the axis-aligned bounding box of this shape.
        """
        if self._bounds is None:
            self._bounds = self._get_bounds()
        return self._bounds.copy()

    @abstractmethod
    def _make_intersect_function(self) -> Callable[[Ray, SurfaceInteraction], bool]:
        """
        Creates a JIT-ed function for intersection tests of this shape.

        Users should not call this method directly. Use the `intersect_function` property instead.
        """
        ...

    @property
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
        if self._intersect_function is None:
            self._intersect_function = self._make_intersect_function()
        return self._intersect_function


class Material(ABC):
    """
    Base class of materials. A material can be emissive, and may scatter a ray on a ray-entity
    intersection.
    """

    def __init__(self):
        self._scatter_function = None

    @abstractmethod
    def _make_scatter_function(self) -> Callable[[Ray, SurfaceInteraction], None]:
        """
        Creates a JIT-ed function for calculations related to this material.

        Users should not call this method directly. Use the `scatter_function` property instead.
        """
        ...

    @property
    def scatter_function(self) -> Callable[[Ray, SurfaceInteraction], None]:
        """
        Returns a JIT-ed function for calculations related to this material.

        * The first argument of the returned function is a ray. Its value should not be changed by
          this function.
        * The second argument of the returned function is a surface interaction. Its `p` and `n`
          fields (which should already been set by a shape instance) are not changed, while its
          `li`, `attenuation`, and `d_out` fields are updated.
        """
        if self._scatter_function is None:
            self._scatter_function = self._make_scatter_function()
        return self._scatter_function


class Entity(ABC):
    """
    Base class of entities. An entity is the combination of consisting shapes and materials.
    """

    def __init__(self):
        self._bounds = None
        self._intersect_function = None

    @abstractmethod
    def _get_bounds(self) -> AlignedBox:
        """
        Calculates the axis-aligned bounding box of this entity.

        Users should not call this method directly. Use the `bounds` property instead.
        """
        ...

    @property
    def bounds(self) -> AlignedBox:
        """
        Returns the axis-aligned bounding box of this entity.
        """
        if self._bounds is None:
            self._bounds = self._get_bounds()
        return self._bounds.copy()

    @abstractmethod
    def _make_intersect_function(self) -> Callable[[Ray, SurfaceInteraction], bool]:
        """
        Creates a JIT-ed function for intersection tests and material calculations related to this
        entity.

        Users should not call this method directly. Use the `intersect_function` property instead.
        """
        ...

    @property
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
        if self._intersect_function is None:
            self._intersect_function = self._make_intersect_function()
        return self._intersect_function


class Integrator(ABC):
    """
    Base class of integrators. An integrator usually takes an entity and then calculates the light
    intensity of each pixel, which directly corresponds to the pixel color in the output image.
    """

    def __init__(self):
        self._integrate_function = None

    @abstractmethod
    def _make_integrate_function(self) -> Callable[[AlignedBox], Spectrum]:
        """
        Creates a JIT-ed function for calculating the light intensity of a given pixel.

        Users should not call this method directly. Use the `integrate_function` property instead.
        """
        ...

    @property
    def integrate_function(self) -> Callable[[AlignedBox], Spectrum]:
        """
        Returns a JIT-ed function for calculating the light intensity of a given pixel.

        * The first (and only) argument of the returned function is an axis-aligned box representing
          the region occupied by this pixel.
        * The return value of the returned function is the light intensity of this pixel.
        """
        if self._integrate_function is None:
            self._integrate_function = self._make_integrate_function()
        return self._integrate_function
