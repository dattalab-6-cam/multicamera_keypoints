import multicam_calibration as mcc
import click
from os.path import join, exists, basename

from o2_utils.selectors import find_files_from_pattern

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
def main(video_dir, board_shape, square_size):

    board_shape = tuple(map(int, board_shape.split("x")))
    camera_names = ["top", "side1", "side2", "side3", "side4", "bottom"]
    video_paths = [find_files_from_pattern(video_dir, f"*.{camera}.mp4") for camera in camera_names]
    video_basename = basename(video_paths[0]).split(".")[0]  # remove the camera name --> 20240422_J04301

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
    save_path = join(video_dir, f"{video_basename}.camera_params.h5")
    mcc.save_calibration(
        all_extrinsics, all_intrinsics, camera_names, save_path, save_format="gimbal"
    )


if __name__ == "__main__":
    main()
