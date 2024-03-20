import os
from os.path import exists, join
from warnings import warn

import click
import h5py
import multicam_calibration as mcc
import numpy as np
import tqdm


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
            "step_dependencies": step_dependencies,
    }

    return triang_config, step_name


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
