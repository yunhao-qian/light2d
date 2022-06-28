"""
Utility functions for the core and other sub-packages.
"""

from typing import Iterable

import numpy as np

from .base import AlignedBox


def aligned_box_union(boxes: Iterable[AlignedBox]) -> AlignedBox:
    union_box = np.empty((2, 2), np.float32)
    union_box[0] = np.inf
    union_box[1] = -np.inf
    for box in boxes:
        union_box[0] = np.min(union_box[0], box[0])
        union_box[1] = np.max(union_box[1], box[1])
    return union_box
