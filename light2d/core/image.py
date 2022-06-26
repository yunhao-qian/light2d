"""
Functions for generating and saving images.
"""

from multiprocessing import Pool

import numpy as np
from numba import njit

from .base import AlignedBox, Entity, F32Array, Integrator


def render(entity: Entity, integrator: Integrator,
           region: tuple[tuple[float, float], tuple[float, float]],
           film_size: tuple[int, int], n_tiles: int = 1) -> F32Array:
    integrate = integrator.integrate_function(entity)
    region = np.array(region, np.float32)

    @njit
    def render_tile(params: tuple[AlignedBox, tuple[int, int], int]) -> F32Array:
        tile_region = params[0]
        tile_size = params[1]
        random_seed = params[2]
        np.random.seed(random_seed)

        x_range = np.linspace(tile_region[0, 0], tile_region[1, 0], tile_size[0] + 1)
        y_range = np.linspace(tile_region[0, 1], tile_region[1, 1], tile_size[1] + 1)
        tile = np.empty((tile_size[1], tile_size[0], 3), np.float32)

        pixel_region = np.empty((2, 2), np.float32)
        for row in range(tile_size[1]):
            pixel_region[0, 1] = y_range[row]
            pixel_region[1, 1] = y_range[row + 1]
            for col in range(tile_size[0]):
                pixel_region[0, 0] = x_range[col]
                pixel_region[1, 0] = x_range[col + 1]
                tile[row, col] = integrate(pixel_region)

        return tile

    if not n_tiles > 1:
        return render_tile((region, film_size,
                            np.random.randint(np.uint32(0), np.uint32(0xFFFFFFFF))))

    tile_width = -(-film_size[0] // n_tiles)
    tile_col_range = np.arange(n_tiles + 1) * tile_width
    tile_col_range[-1] = film_size[0]
    tile_x_range = (film_size[0] - tile_col_range).astype(np.float32) / np.float32(film_size[0]) * \
        region[0, 0] + tile_col_range.astype(np.float32) / np.float32(film_size[0]) * region[1, 0]

    tile_height = -(-film_size[1] // n_tiles)
    tile_row_range = np.arange(n_tiles + 1) * tile_height
    tile_row_range[-1] = film_size[1]
    tile_y_range = (film_size[1] - tile_row_range).astype(np.float32) / np.float32(film_size[1]) * \
        region[0, 1] + tile_row_range.astype(np.float32) / np.float32(film_size[1]) * region[1, 1]

    process_args = []
    tile_indices = []
    for i in range(n_tiles):
        y_min = tile_y_range[i]
        y_max = tile_y_range[i + 1]
        row_min = tile_row_range[i]
        row_max = tile_row_range[i + 1]
        for j in range(n_tiles):
            x_min = tile_x_range[j]
            x_max = tile_x_range[j + 1]
            col_min = tile_col_range[j]
            col_max = tile_col_range[j + 1]

            process_args.append((
                np.array(((x_min, y_min), (x_max, y_max))),
                (col_max - col_min, row_max - row_min),
                np.random.randint(np.uint32(0), np.uint32(0xFFFFFFFF)),
            ))
            tile_indices.append(((row_min, row_max), (col_min, col_max)))

    with Pool(np.square(n_tiles)) as pool:
        tiles = pool.map(render_tile, process_args)
    film = np.empty((film_size[1], film_size[0], 3), np.float32)
    for tile, ((row_min, row_max), (col_min, col_max)) in zip(tiles, tile_indices):
        film[row_min:row_max, col_min:col_max] = tile
    return film
