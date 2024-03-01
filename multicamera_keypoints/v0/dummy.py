import numpy as np
import sys
import os

from multicamera_keypoints.vid_utils import count_frames


def save_array(array, vid_path, name):
    fullfile, ext = os.path.splitext(vid_path)
    out_file = fullfile + "." + name + ".npy"
    if not os.path.exists(out_file):
        np.save(out_file, array)


def main(vid_path):
    nframes = count_frames(vid_path)
    arr = np.arange(nframes)
    save_array(arr, vid_path, "dummy")


if __name__ == "__main__":
    vid_path = sys.argv[1]
    main(vid_path)
    exit()
