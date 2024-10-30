import os
from os.path import exists, join
from warnings import warn

import click
import h5py
import multicam_calibration as mcc
import numpy as np
import pandas as pd
import tqdm

from o2_utils.selectors import find_files_from_pattern


def make_config(
    PACKAGE_DIR,
    sec_per_frame=0.02,
    output_name_suffix=None,
    step_dependencies=None,
):
    """Create a default config for the TRIANGULATION step.

    Parameters
    ----------
    PACKAGE_DIR : str
        The directory where the package is installed.

    sec_per_frame : float, optional
        The number of seconds per frame for the triangulation step. The default is 0.02.

    output_name_suffix : str, optional
        The suffix to add to the output name. The default is None.
        Example: "v2" --> "robust_triangulation.v2.npy", and the step name
        will be "TRIANGULATION.v2".

        NB: currently no reason you'd do this with the triang step, as there's no
        neural network involved.

    step_dependencies : list, optional
        The list of step names for the dependencies of this step. The default is ["HRNET"].
        These steps will be checked for completion before running this step.

    Returns
    -------
    triang_config : dict
        The configuration for the triangulation step.

    step_name : str
        The name of the triangulation step. (default: "TRIANGULATION")
    """
    if output_name_suffix is not None:
        output_name = f"robust_triangulation.{output_name_suffix}.npy"
        step_name = f"TRIANGULATION.{output_name_suffix}"
    else:
        output_name = "robust_triangulation.npy"
        step_name = "TRIANGULATION"

    if step_dependencies is None:
        step_dependencies = ["HRNET"]

    triang_config = {
        "slurm_params": {
            "mem": "6GB",
            "gpu": False,
            "sec_per_frame": sec_per_frame,
            "ncpus": 1,
            "jobs_in_progress": {},
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "triangulation.py"),
            "conda_env": "dataPy_NWB2",  # TODO: make this dynamic?
        },
        "func_args": {  # NB: these args **must** be in the right order here.
            "video_dir": "{video_dir}",
            "calib_file": "{calib_file}",
        },
        "output_info": {
            "output_name": output_name,
        },
        "step_dependencies": [],  # TODO: doesn't really work to put HRNET here, since HRNET is at hte video level and this is at the session level.
        "pipeline_info": {
            "processing_level": "session",
        },
    }

    return triang_config, step_name


@click.command()
@click.argument("vid_dir")
@click.argument("calib_file")
@click.option("--conf_threshold", default=0.25, help="Confidence threshold for keypoint detection, below which detections are ignored for the triangulation step.")
@click.option("--output_name", default="robust_triangulation.npy", help="Suffix to be appended to the name of the video to make the output file name.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(vid_dir, calib_file, conf_threshold=0.25, output_name="robust_triangulation.npy", overwrite=False):
    
    # Report start to user
    print(f"Triangulating vids in dir {vid_dir} with calib file {calib_file}...")

    # Find the keypoint files in the directory
    kp_files = find_files_from_pattern(vid_dir, "*.keypoints.h5", n_expected=-1)
    if len(kp_files) == 0:
        print(f"No keypoint files found in {vid_dir}, exiting.")
        return
    
    # Prepare output info
    session_name = os.path.basename(kp_files[0]).split(".")[0]
    assert output_name.endswith(".npy"), "Output name must end with .npy"
    out_name = join(vid_dir, session_name + "." + output_name)
    print(out_name)

    # Stop if file exists
    if exists(out_name) and not overwrite:
        print(f"{os.path.basename(out_name)} exists, exiting...")
        return

    # Load the calibration data
    all_extrinsics, all_intrinsics, camera_names = mcc.load_calibration(
        calib_file, "gimbal"
    )

    # Match kp files to camera names
    kp_files = [[f for f in kp_files if c in f][0] for c in camera_names]
    print(camera_names, kp_files)

    # Check if frame alignment is required, if so look for alignment file in the video dir
    alignment_file = join(vid_dir, "aligned_frame_numbers.csv")
    if not exists(alignment_file):
        print(f"Assuming all videos have the same number of frames. If this is not the case, please provide an alignment file at {alignment_file}.")
        do_alignment = False
    else:
        align_df = pd.read_csv(alignment_file)  # cols are top, bottom, side1, ..., side4, trigger_number
        max_n_frames = align_df.shape[0]
        do_alignment = True

    # Prep the data
    all_uvs = []
    for cam, kp_file in zip(camera_names, kp_files):
        try:
            with h5py.File(kp_file, "r") as h5:
                uvs = h5["uv"][()]
                confs = h5["conf"][()]

                # Replace low confidence detections with nan
                uvs[confs < conf_threshold] = np.nan  

                # Align the frames if necessary, setting missing values to nan
                if do_alignment:
                    aligned_confs = np.nan * np.zeros((max_n_frames, confs.shape[1]))
                    aligned_uvs = np.nan * np.zeros((max_n_frames, uvs.shape[1], uvs.shape[2]))
                    align_vec = align_df[cam].values
                    aligned_confs[~pd.isnull(align_vec), ...] = confs
                    aligned_uvs[~pd.isnull(align_vec), ...] = uvs
                    all_uvs.append(aligned_uvs)
                else:
                    all_uvs.append(uvs)

        except OSError:
            warn(
                f"{kp_file} could not be loaded! Probably h5 file was not closed properly due to a job timing out."
            )
            print("Exiting.")
            return

    all_uvs = np.array(all_uvs)  # shape: (n_cams, n_frames, n_keypoints, 2)

    # Do the triangulation
    poses = []
    for i in tqdm.tqdm(range(len(all_uvs[0]))):  # ie for each frame
        pts = mcc.triangulate(all_uvs[:, i], all_extrinsics, all_intrinsics)  # pass in uvs of shape (n_cams, n_keypoints, 2)
        poses.append(pts)

    # Save the results
    np.save(out_name, np.array(poses))

    print(f"Triangulated poses saved at {out_name}!")

    return


if __name__ == "__main__":
    main()
