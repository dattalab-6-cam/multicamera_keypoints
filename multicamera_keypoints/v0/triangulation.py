import os
from os.path import exists, join
from warnings import warn

import click
import h5py
import multicam_calibration as mcc
import numpy as np
import tqdm


@click.command()
@click.argument("vid_dir")
@click.argument("calib_file")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(vid_dir, calib_file, overwrite=False):
    print(f"Triangulating vids in dir {vid_dir} with calib file {calib_file}...")

    all_extrinsics, all_intrinsics, camera_names = mcc.load_calibration(
        calib_file, "gimbal"
    )

    session_name = os.path.basename(vid_dir)
    out_name = join(vid_dir, session_name + ".robust_triangulation.npy")
    print(out_name)

    # Stop if file exists
    if exists(out_name) and not overwrite:
        print(f"{os.path.basename(out_name)} exists, exiting...")
        return

    # Prep the data
    all_uvs = []
    for c in camera_names:
        kp_file = join(vid_dir, session_name + f".{c}.keypoints.h5")
        try:
            with h5py.File(kp_file, "r") as h5:
                uvs = h5["uv"][()]
                confs = h5["conf"][()]
                uvs[confs < 0.25] = np.nan  # remove low confidence detections
                all_uvs.append(uvs)
        except OSError:
            warn(
                f"{kp_file} could not be loaded! Probably h5 file was not closed properly due to a job timing out."
            )
            print("Exiting.")
            return

    all_uvs = np.array(all_uvs)

    # Do the triangulation
    poses = []
    for i in tqdm.tqdm(range(len(all_uvs[0]))):
        pts = mcc.triangulate(all_uvs[:, i], all_extrinsics, all_intrinsics)
        poses.append(pts)

    # Save the results
    np.save(out_name, np.array(poses))

    print(f"Triangulated poses saved at {out_name}!")

    return


if __name__ == "__main__":
    main()
