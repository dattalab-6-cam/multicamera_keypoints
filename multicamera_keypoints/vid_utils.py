import numpy as np
import av


def count_frames(file_name):
    with av.open(file_name, 'r') as reader:
        return reader.streams.video[0].frames
