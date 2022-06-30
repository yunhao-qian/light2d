"""
Definition of the path tracer integrator.
"""

from typing import Callable

import numpy as np
from numba import njit

from ..core.base import EPSILON, AlignedBox, Entity, Integrator, Ray, Spectrum, SurfaceInteraction


class PathTracer(Integrator):
    """
    Path tracer is an integrator using the path tracing algorithm.

    The integrator uses stratified sampling in all of the x, y, and angle coordinates. The region
    occupied by the pixel is split into a `n_samples`-by-`n_samples` grid, and the angle range
    `[0, 2 pi)` is split into `n_samples ^ 2` intervals. Each ray has an origin within one of the
    grid cells and a direction within one of the angle intervals (without repetition). Therefore,
    there are `n_samples ^ 2` samples in total.

    For an individual light path, the integrator guarantees to trace for the first `n_steps` (as
    long as the ray is scattered by the materials). After that, for every additional step the
    integrator may stop tracing by a probability of `russian_roulette_q`. The integrator is then
    able to calculate an unbiased estimate of the sum of light intensities along the light path.
    """

    def __init__(self, entity: Entity, n_samples: int, n_steps: int = 3,
                 russian_roulette_q: float = 0.05):
        """
        Creates a path tracer for the given entity with the specified parameters. See the class
        documentation for details about the parameters.
        """
        self.entity = entity
        self.n_samples = np.uint32(n_samples)
        self.n_steps = np.uint32(n_steps)
        self.russian_roulette_q = np.float32(russian_roulette_q)

    @property
    def integrate_function(self) -> Callable[[AlignedBox], Spectrum]:
        n_samples = self.n_samples
        n_steps = self.n_steps
        russian_roulette_q = self.russian_roulette_q
        entity_intersect = self.entity.intersect_function

        @njit
        def get_scattered_ray(ray: Ray, interaction: SurfaceInteraction) -> None:
            p = interaction[0:2]
            n = interaction[2:4]
            d_out = interaction[10:12]
            n_offset = n * (EPSILON / np.linalg.norm(n))
            if np.dot(d_out, n) < 0:
                o_out = p - n_offset
            else:
                o_out = p + n_offset
            ray[0:2] = o_out
            ray[2:4] = d_out
            ray[4] = np.inf

        @njit
        def trace(ray: Ray) -> Spectrum:
            interaction = np.empty(12, np.float32)
            li_sum = np.zeros(3, np.float32)
            net_attenuation = np.ones(3, np.float32)

            for _ in range(n_steps):
                if not entity_intersect(ray, interaction):
                    return li_sum
                li = interaction[4:7]
                li_sum += li * net_attenuation
                attenuation = interaction[7:10]
                if not np.any(attenuation > 0):
                    return li_sum
                net_attenuation *= attenuation
                get_scattered_ray(ray, interaction)

            while True:
                if not np.float32(np.random.uniform(0, 1)) < russian_roulette_q or \
                   not entity_intersect(ray, interaction):
                    return li_sum
                li = interaction[4:7]
                net_attenuation /= 1 - russian_roulette_q
                li_sum += li * net_attenuation
                attenuation = interaction[7:10]
                if not np.any(attenuation > 0):
                    return li_sum
                net_attenuation *= attenuation
                get_scattered_ray(ray, interaction)

        @njit
        def integrate(region: AlignedBox) -> Spectrum:
            x_range = np.linspace(region[0, 0], region[1, 0], n_samples + 1)
            y_range = np.linspace(region[0, 1], region[1, 1], n_samples + 1)
            angle_range = np.linspace(np.float32(0), np.float32(np.pi * 2),
                                      np.square(n_samples) + 1)
            angle_order = np.arange(np.int32(np.square(n_samples)))
            np.random.shuffle(angle_order)

            k = np.uint32(0)
            ray = np.empty(5, np.float32)
            li_sum = np.zeros(3, np.float32)
            valid_count = np.uint32(0)

            for row in range(n_samples):
                y_min = y_range[row]
                y_max = y_range[row + 1]
                for col in range(n_samples):
                    x_min = x_range[col]
                    x_max = x_range[col + 1]
                    i_angle = angle_order[k]
                    k += np.uint32(1)
                    angle_min = angle_range[i_angle]
                    angle_max = angle_range[i_angle + 1]

                    ray[0] = np.random.uniform(x_min, x_max)
                    ray[1] = np.random.uniform(y_min, y_max)
                    angle = np.float32(np.random.uniform(angle_min, angle_max))
                    ray[2] = np.cos(angle)
                    ray[3] = np.sin(angle)
                    ray[4] = np.inf

                    li = trace(ray)
                    if np.all(np.isfinite(li)):
                        li_sum += li
                        valid_count += np.uint32(1)

            return li_sum / np.float32(valid_count)

        return integrate
