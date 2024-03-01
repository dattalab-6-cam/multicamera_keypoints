from functools import lru_cache

import av
import numpy as np


def count_frames(file_name):
    """Count the number of frames in a video file."""
    with av.open(file_name, "r") as reader:
        return reader.streams.video[0].frames


@lru_cache
def count_frames_cached(file_name):
    """Count the number of frames in a video file."""
    with av.open(file_name, "r") as reader:
        return reader.streams.video[0].frames


def crop_image(image, centroid, crop_size):
    """Crop an image around a centroid.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to crop.

    centroid: tuple of int
        (x,y) coordinates of the centroid.

    crop_size: int or tuple(int,int)
        Size of the crop around the centroid. Either a single int for a square
        crop, or a tuple of ints (w,h) for a rectangular crop.

    Returns
    -------
    image: ndarray of shape (crop_size, crop_size, 3)
        Cropped image.
    """
    if isinstance(crop_size, tuple):
        w, h = crop_size
    else:
        w, h = crop_size, crop_size
    x, y = int(centroid[0]), int(centroid[1])

    x_min = max(0, x - w // 2)
    y_min = max(0, y - h // 2)
    x_max = min(image.shape[1], x + w // 2)
    y_max = min(image.shape[0], y + h // 2)

    cropped = image[y_min:y_max, x_min:x_max]
    padded = np.zeros((h, w, *image.shape[2:]), dtype=image.dtype)
    pad_x = max(w // 2 - x, 0)
    pad_y = max(h // 2 - y, 0)
    padded[pad_y : pad_y + cropped.shape[0], pad_x : pad_x + cropped.shape[1]] = cropped
    return padded
