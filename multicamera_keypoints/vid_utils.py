import numpy as np
import av


def count_frames(file_name):
    """Count the number of frames in a video file.
    """
    with av.open(file_name, 'r') as reader:
        return reader.streams.video[0].frames
