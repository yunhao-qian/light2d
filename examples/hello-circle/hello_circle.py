
import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '../..'))

import light2d
from light2d import entities, integrators, materials, shapes


if __name__ == '__main__':
    film = light2d.render(
        integrator=integrators.PathTracer(
            entity=entities.SimpleEntity(
                shape=shapes.Circle(
                    center=(0, 0),
                    radius=1,
                ),
                material=materials.ConstantLight(
                    li=(0.6, 0.8, 1.0),
                ),
            ),
            n_samples=16,
        ),
        region=((-2, -2), (2, 2)),
        film_size=(512, 512),
        n_tiles=6,
    )

    light2d.save(film, os.path.join(file_dir, 'hello_circle.png'))
