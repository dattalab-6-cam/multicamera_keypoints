import multicam_calibration as mcc
import matplotlib.pyplot as plt
import numpy as np
import joblib
import h5py
import glob
import tqdm
from os.path import join, exists
import os
from warnings import warn
from moseq_fo.util.file_utils import find_file_from_pattern
import sys


def main(vid_path, calib_file):
    
    print(f'vid_path: {vid_path}')
    print(f'calib_path: {calib_file}')

    all_extrinsics, all_intrinsics, camera_names = mcc.load_calibration(calib_file, 'gimbal')
#     suffix = '.COMPRESSED'
    suffix = ''
    fname = join(vid_path, 'robust_triangulation' + suffix + '.npy')

    # Stop if file exists
    if exists(fname): 
        print(f'{os.path.basename(fname)} exists, exiting...')
        return
    
    # Prep the data
    all_uvs = []
    for c in camera_names:
        kp_file = join(vid_path, c + suffix + '.keypoints.h5')
        try:
            with h5py.File(kp_file,'r') as h5:
                uvs = h5['uv'][()]
                confs = h5['conf'][()]
                uvs[confs < 0.25] = np.nan  # remove low confidence detections
                all_uvs.append(uvs)
        except OSError:
            warn(f'{kp_file} could not be loaded! Probably file was not closed properly due to a job timing out.')
            print('Exiting.')
            return
            
    all_uvs = np.array(all_uvs)

    # Do the triangulation
    poses = []
    for i in tqdm.tqdm(range(len(all_uvs[0]))):
        pts = mcc.triangulate(all_uvs[:,i], all_extrinsics, all_intrinsics)
        poses.append(pts)

    # Save the results
    np.save(fname, np.array(poses))

    return



if __name__ == "__main__":
    vid_path = sys.argv[1]
    calib_file = sys.argv[2]
    main(vid_path, calib_file)