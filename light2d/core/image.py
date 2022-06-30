"""
Functions for generating and saving images.
"""

from multiprocessing import Pool

import numpy as np
from numba import njit
from PIL import Image

from .base import AlignedBox, Entity, F32Array, Integrator


def render(integrator: Integrator, region: tuple[tuple[float, float], tuple[float, float]],
           film_size: tuple[int, int], n_tiles: int = 1) -> F32Array:
    """
    Renders an image of the given entity with the specified parameters.

    * `integrator` is the integrator used to calculate light intensities of pixels.
    * `region` specified the region (in the entity's coordinate system) to be rendered. The first
      element of the tuple is the minimum x and y coordinates, and the second element is the maximum
      x and y coordinates.
    * `film_size` is the width and height of the film. It should have the same aspect ratio as the
      specified region.
    * `n_tiles` controls the multiprocessing setting. If this value is not greater than 1, rendering
      is performed on a single process. Otherwise, if this value is greater than 1, the image is
      uniformly split into a `n_tiles`-by-`n_tiles` grid of tiles, where each tile is rendered on a
      separate process. Therefore, there are `n_tiles ^ 2` rendering processes in total.
    * The return value is the rendered image represented by a float32 array of shape
      `(film_size[1], film_size[0], 3)`.
    """
    integrate = integrator.integrate_function
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


def save(film: F32Array, filename: str, gamma: float = 2.2) -> None:
    """
    Saves the rendered image into an image file.

    * `film` is the rendered image.
    * `filename` is the name of the image file.
    * `gamma` is used for gamma correction. The color space of the saved image is assumed to be
      sRGB, so the default gamma value is 2.2. Set this value to 1 if no gamma correction should be
      performed.
    """
    film = np.power(film, np.float32(1 / gamma))
    film *= 255
    np.clip(film, 0, 255, out=film)
    film = np.flipud(film.astype(np.uint8))
    Image.fromarray(film, 'RGB').save(filename)
