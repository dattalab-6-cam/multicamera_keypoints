from os.path import basename, join, exists

import click
import multicam_calibration as mcc
from o2_utils.selectors import find_files_from_pattern


def make_config(
    PACKAGE_DIR, 
    sec_per_frame=1.1,
    output_name_suffix=None,
    step_dependencies=None,
):
    """Create a default config for the calibration step.

    Parameters
    ----------
    PACKAGE_DIR : str
        The directory where the package is installed.

    sec_per_frame : float, optional
        The number of seconds per frame for the calibration step. The default is 1.1.
        1.1 = (0.15 * 6) + 0.2  # 0.13 s/fr is a conservative estimate for detection for 6 workers for one vid, times 6 vids per calibration, plus extra time for the extra steps of calibration

    output_name_suffix : str, optional
        The suffix to add to the output name. The default is None.
        Example: "v2" --> "camera_params.v2.h5", and the step name 
        will be "CALIBRATION.v2".

    step_dependencies : list, optional
        The list of step names for the dependencies of this step. The default is None.
        These steps will be checked for completion before running this step.

    Returns
    -------
    calib_config : dict
        The configuration for the calibration step. 

    step_name : str
        The name of the calibration step. (default: "CALIBRATION")
    """
    if output_name_suffix is not None:
        output_name = f"camera_params.{output_name_suffix}.h5"
        step_name = f"CALIBRATION.{output_name_suffix}"
    else:
        output_name = "camera_params.h5"
        step_name = "CALIBRATION"

    calib_config = {
        "slurm_params": {
            "mem": "24GB",
            "gpu": False,
            "sec_per_frame": sec_per_frame,
            "ncpus": 6,
            "jobs_in_progress": {},
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "calibration.py"),
            "conda_env": "dataPy_NWB2",  # TODO: make this dynamic
        },
        "func_args": {
            "video_dir": "{video_dir}",  # TODO: get these func args from a more reasonable location
        },
        "output_info": {
            "output_name": output_name,  # saves an h5 file
        },
    }

    if step_dependencies is not None:
        calib_config["step_dependencies"] = step_dependencies
    else:
        calib_config["step_dependencies"] = []

    return calib_config, step_name


@click.command()
@click.argument("video_dir")
@click.option(
    "--board-shape",
    type=str,
    default="5x7",
    help='Number of internal corners in the chessboard, as a string like "5x7".',
)
@click.option(
    "--square-size", default=12.5, help="Size of the squares in the chessboard"
)
@click.option("--output_name", default="camera_params.h5", help="Suffix appended to the video name to create the output file.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing calibration data.")
def main(video_dir, board_shape, square_size, output_name="camera_params.h5", overwrite=False):

    board_shape = tuple(map(int, board_shape.split("x")))
    camera_names = ["top", "side1", "side2", "side3", "side4", "bottom"]
    video_paths = [find_files_from_pattern(video_dir, f"*.{camera}.mp4") for camera in camera_names]
    video_basename = basename(video_paths[0]).split(".")[0]  # remove the camera name --> 20240422_J04301

    # Check for potential overwriting 
    save_path = join(video_dir, f"{video_basename}.{output_name}")
    if not overwrite and exists(save_path):
        print(f"Calibration already exists at {save_path}. Exiting.")
        return

    # detect calibration object in each video
    all_calib_uvs, all_img_sizes = mcc.run_calibration_detection(
        video_paths,
        mcc.detect_chessboard,
        n_workers=15,
        overwrite=False,
        detection_options=dict(
            board_shape=board_shape, scale_factor=1, match_score_min_diff=0.1
        ),
    )

    # plot corner-match scores for each frame
    fig = mcc.plot_chessboard_qc_data(video_paths)
    fig.savefig(join(video_dir, f"{video_basename}.qc.png"))

    # display a table with the detections shared between camera pairs
    table = mcc.summarize_detections(all_calib_uvs)
    print(table)

    ### INITIAL

    calib_objpoints = mcc.generate_chessboard_objpoints(board_shape, square_size)

    all_extrinsics, all_intrinsics, calib_poses = mcc.calibrate(
        all_calib_uvs,
        all_img_sizes,
        calib_objpoints,
        root=0,
        n_samples_for_intrinsics=100,
    )

    (
        fig,
        mean_squared_error,
        reprojections,
        transformed_reprojections,
    ) = mcc.plot_residuals(
        all_calib_uvs, all_extrinsics, all_intrinsics, calib_objpoints, calib_poses
    )
    fig.savefig(join(video_dir, f"{video_basename}.initial_residuals.png"))

    #### BUNDLE ADJUSTMENT

    (
        adj_extrinsics,
        adj_intrinsics,
        adj_calib_poses,
        use_frames,
        result,
    ) = mcc.bundle_adjust(
        all_calib_uvs,
        all_extrinsics,
        all_intrinsics,
        calib_objpoints,
        calib_poses,
        n_frames=5000,
        ftol=1e-4,
    )

    fig, median_error, reprojections, transformed_reprojections = mcc.plot_residuals(
        all_calib_uvs[:, use_frames],
        adj_extrinsics,
        adj_intrinsics,
        calib_objpoints,
        adj_calib_poses,
    )

    fig.savefig(join(video_dir, f"{video_basename}.bundle_adjusted_residuals.png"))

    # save for GIMBAL
    mcc.save_calibration(
        all_extrinsics, all_intrinsics, camera_names, save_path, save_format="gimbal"
    )


if __name__ == "__main__":
    main()
