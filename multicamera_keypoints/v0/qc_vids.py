# import gimbal
import os
from os.path import exists, join

import click
import cv2
import h5py
import imageio
import multicam_calibration as mcc
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from keypoint_moseq.util import get_edges
# from keypoint_moseq.viz import crop_image, overlay_keypoints_on_image
from scipy.ndimage import gaussian_filter1d, median_filter
from tqdm.auto import tqdm
from textwrap import fill
from vidio.read import OpenCVReader

from multicamera_keypoints.io import load_config


def add_text(im, text, position):
    im = cv2.putText(
        im,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return im


def build_node_hierarchy(bodyparts, skeleton, root_node):
    """
    Define a rooted hierarchy based on the edges of a spanning tree.

    Parameters
    ----------
    bodyparts: list of str
        Ordered list of node names.

    skeleton: list of tuples
        Edges of the spanning tree as pairs of node names.

    root_node: str
        The desired root node of the hierarchy

    Returns
    -------
    node_order: array of shape (num_nodes,)
        Integer array specifying an ordering of nodes in which parents
        precede children (i.e. a topological ordering).

    parents: array of shape (num_nodes,)
        Child-parent relationships using the indexes from `node_order`,
        such that `parent[i]==j` when `node_order[j]` is the parent of
        `node_order[i]`.

    Raises
    ------
    ValueError
        The edges in `skeleton` do not define a spanning tree.
    """
    G = nx.Graph()
    G.add_nodes_from(bodyparts)
    G.add_edges_from(skeleton)

    if not nx.is_tree(G):
        cycles = list(nx.cycle_basis(G))
        raise ValueError(
            "The skeleton does not define a spanning tree, "
            "as it contains the following cycles: {}".format(cycles)
        )

    if not nx.is_connected(G):
        raise ValueError(
            "The skeleton does not define a spanning tree, "
            "as it contains multiple connected components."
        )

    node_order = list(nx.dfs_preorder_nodes(G, root_node))
    parents = np.zeros(len(node_order), dtype=int)

    for i, j in skeleton:
        i, j = node_order.index(i), node_order.index(j)
        if i < j:
            parents[j] = i
        else:
            parents[i] = j

    node_order = np.array([bodyparts.index(n) for n in node_order])
    return node_order, parents


def get_edges(use_bodyparts, skeleton):
    """Represent the skeleton as a list of index-pairs.

    Parameters
    -------
    use_bodyparts: list
        Bodypart names

    skeleton: list
        Pairs of bodypart names as tuples (bodypart1,bodypart2)

    Returns
    -------
    edges: list
        Pairs of indexes representing the enties of `skeleton`
    """
    edges = []
    if len(skeleton) > 0:
        if isinstance(skeleton[0][0], int):
            edges = skeleton
        else:
            assert use_bodyparts is not None, fill(
                "If skeleton edges are specified using bodypart names, "
                "`use_bodyparts` must be specified"
            )

            for bp1, bp2 in skeleton:
                if bp1 in use_bodyparts and bp2 in use_bodyparts:
                    edges.append(
                        [use_bodyparts.index(bp1), use_bodyparts.index(bp2)]
                    )
    return edges


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


def overlay_keypoints_on_image(
    image,
    coordinates,
    edges=[],
    keypoint_colormap="autumn",
    keypoint_colors=None,
    node_size=5,
    line_width=2,
    copy=False,
    opacity=1.0,
):
    """Overlay keypoints on an image.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to overlay keypoints on.

    coordinates: ndarray of shape (num_keypoints, 2)
        Array of keypoint coordinates.

    edges: list of tuples, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of indexes.

    keypoint_colormap: str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints.

    keypoint_colors : array-like, shape=(num_keypoints,3), default=None
        Color for each keypoint. If None, the keypoint colormap is used.
        If the dtype is int, the values are assumed to be in the range 0-255,
        otherwise they are assumed to be in the range 0-1.

    node_size: int, default=5
        Size of the keypoints.

    line_width: int, default=2
        Width of the skeleton lines.

    copy: bool, default=False
        Whether to copy the image before overlaying keypoints.

    opacity: float, default=1.0
        Opacity of the overlay graphics (0.0-1.0).

    Returns
    -------
    image: ndarray of shape (height, width, 3)
        Image with keypoints overlayed.
    """
    if copy or opacity < 1.0:
        canvas = image.copy()
    else:
        canvas = image

    if keypoint_colors is None:
        cmap = plt.colormaps[keypoint_colormap]
        colors = np.array(cmap(np.linspace(0, 1, coordinates.shape[0])))[:, :3]
    else:
        colors = np.array(keypoint_colors)

    if isinstance(colors[0, 0], float):
        colors = [tuple([int(c) for c in cs * 255]) for cs in colors]

    # overlay skeleton
    for i, j in edges:
        if np.isnan(coordinates[i, 0]) or np.isnan(coordinates[j, 0]):
            continue
        pos1 = (int(coordinates[i, 0]), int(coordinates[i, 1]))
        pos2 = (int(coordinates[j, 0]), int(coordinates[j, 1]))
        canvas = cv2.line(canvas, pos1, pos2, colors[i], line_width, cv2.LINE_AA)

    # overlay keypoints
    for i, (x, y) in enumerate(coordinates):
        if np.isnan(x) or np.isnan(y):
            continue
        pos = (int(x), int(y))
        canvas = cv2.circle(canvas, pos, node_size, colors[i], -1, lineType=cv2.LINE_AA)

    if opacity < 1.0:
        image = cv2.addWeighted(image, 1 - opacity, canvas, opacity, 0)
    return image


# def load_config(project_dir, check_if_valid=True):
#     """Load a project config file.

#     Parameters
#     ----------
#     project_dir: str
#         Directory containing the config file

#     check_if_valid: bool, default=True
#         Check if the config is valid using
#         :py:func:`multicamera_keypoints.v0.io.check_config_validity`

#     Returns
#     -------
#     config: dict
#     """
#     config_path = os.path.join(project_dir, "keypoint_config.yml")

#     with open(config_path, "r") as stream:
#         config = yaml.safe_load(stream)

#     return config


# this is the order that we get keypoints from the hrnet
bodyparts_hrnet_ordering = [
    "tail_tip",
    "tail_base",
    "spine_low",
    "spine_mid",
    "spine_high",
    "left_ear",
    "right_ear",
    "forehead",
    "nose_tip",
    "left_hind_paw_front",
    "left_hind_paw_back",
    "right_hind_paw_front",
    "right_hind_paw_back",
    "left_fore_paw",
    "right_fore_paw",
]

# When we pass the keypoints to the gimbal, we need to reorder them
# in order to respect the "node hierarchy" that gimbal requires.
# This is the order that they come out of gimbal in.
bodyparts_gimbal_ordering = [
    "spine_low",
    "tail_base",
    "spine_mid",
    "spine_high",
    "left_ear",
    "right_ear",
    "forehead",
    "nose_tip",
    "left_fore_paw",
    "right_fore_paw",
    "left_hind_paw_back",
    "left_hind_paw_front",
    "right_hind_paw_back",
    "right_hind_paw_front",
]

skeleton = [
    ["tail_base", "spine_low"],
    ["spine_low", "spine_mid"],
    ["spine_mid", "spine_high"],
    ["spine_high", "left_ear"],
    ["spine_high", "right_ear"],
    ["spine_high", "forehead"],
    ["forehead", "nose_tip"],
    ["left_hind_paw_back", "left_hind_paw_front"],
    ["spine_low", "left_hind_paw_back"],
    ["right_hind_paw_back", "right_hind_paw_front"],
    ["spine_low", "right_hind_paw_back"],
    ["spine_high", "left_fore_paw"],
    ["spine_high", "right_fore_paw"],
]


use_bodyparts = bodyparts_hrnet_ordering[1:] # we exclude the tail tip from downstream analysis
use_bodyparts_ix = np.array(
    [bodyparts_hrnet_ordering.index(bp) for bp in use_bodyparts]
)
edges = np.array(get_edges(use_bodyparts, skeleton))
node_order, parents = build_node_hierarchy(use_bodyparts, skeleton, "spine_low")
edges = np.argsort(node_order)[edges]


def make_config(
    PACKAGE_DIR,
    project_dir,
    sec_per_frame=0.25,
    nframes_to_show=7200,
    output_name_suffix=None,
    step_dependencies=None,
):
    """Create a default config for the QCVID step.

    Parameters
    ----------
    PACKAGE_DIR : str
        The directory where the package is installed.

    sec_per_frame : float, optional
        The number of seconds per frame for the quality control videos step. 
        The default is 0.25 (it's quite slow!).

    step_dependencies : list, optional
        The list of step names for the dependencies of this step. The default is ["GIMBAL"].
        These steps will be checked for completion before running this step.

    Returns
    -------
    qc_config : dict
        The configuration for the qc videos step.

    step_name : str
        The name of the step. (default: "QCVID")
    """

    if step_dependencies is None:
        step_dependencies = ["GIMBAL"]

    step_name = "QCVID"

    qc_config = {
        "slurm_params": {
            "mem": "16GB",
            "gpu": False,
            "sec_per_frame": sec_per_frame,
            "ncpus": 1,
            "jobs_in_progress": {},
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "qc_vids.py"),
            "conda_env": "dataPy_torch2",  # TODO: make this dynamic
            "modules": ["gcc/9.2.0", "ffmpeg"],
        },
        "func_args": {  # NB: these args **must** be in the right order here.
            "project_dir": f"{project_dir}",
            "session_name": "{session_name}",
        },
        "func_kwargs": {
            "nframes": nframes_to_show,
        },
        "output_info": {
        },
        "step_dependencies": step_dependencies,
        "pipeline_info": {
            "processing_level": "session",
        },
    }

    return qc_config, step_name


def load_hrnet_uvs(video_dir, video_names, output_name, conf_threshold=0.25, align_df=None):
    """Load HRNET detections for a set of videos.

    Parameters
    ----------
    video_dir : str
        The directory containing the videos.

    video_names : list of str
        The names of the videos to load detections for.

    output_name : str
        The name of the output file to load.

    conf_threshold : float, optional
        The confidence threshold to apply to the detections. The default is 0.25.

    align_df : pd.DataFrame, optional
        A DataFrame containing the alignment information for the videos. The default is None.

    Returns
    -------
    detections : dict
        A dictionary of detections for each video.
    """
    detections = {}
    for v in video_names:
        camera = v.split(".")[1]
        detn_file = join(video_dir, v + "." + output_name)
        if not exists(detn_file):
            detections[v] = None
            continue

        with h5py.File(detn_file, "r") as h5:
            uvs = h5["uv"][()][:, use_bodyparts_ix][:, node_order]
            confs = h5["conf"][()][:, use_bodyparts_ix][:, node_order]
            mask = confs < conf_threshold
            uvs[mask] = np.nan
        
        if align_df is not None:
            max_n_frames = align_df.shape[0]
            # aligned_confs = np.nan * np.zeros((max_n_frames, confs.shape[1]))
            aligned_uvs = np.nan * np.zeros((max_n_frames, uvs.shape[1], uvs.shape[2]))
            align_vec = align_df[camera].values
            aligned_uvs[~pd.isnull(align_vec), ...] = uvs
            detections[camera] = aligned_uvs
        else:
            detections[camera] = uvs

    return detections


def load_triangulation_uvs(video_dir, session_name, all_extrinsics, all_intrinsics, camera_names):
    """Load triangulation uvs for a session.

    Parameters
    ----------
    video_dir : str
        The directory containing the videos.

    session_name : str
        The name of the session.

    all_extrinsics : list of ndarray
        The extrinsics for each camera.

    all_intrinsics : list of ndarray
        The intrinsics for each camera.

    Returns
    -------
    triang_positions : ndarray
        The triangulated positions for the session.
    """
    triang_file = join(video_dir, session_name + ".robust_triangulation.npy")
    if not exists(triang_file):
        return None

    triang_positions = np.load(triang_file)
    triang_positions = triang_positions[:, use_bodyparts_ix][:, node_order]

    # Project the triangulated positions to each camera
    triang_uvs = [
        mcc.project_points(triang_positions, ext, *intr)
        for ext, intr in zip(all_extrinsics, all_intrinsics)
    ]

    triang_uvs = {cam: uv for cam, uv in zip(camera_names, triang_uvs)}

    return triang_uvs


def load_gimbal_uvs(video_dir, session_name, all_extrinsics, all_intrinsics, camera_names):
    """Load gimbal positions for a session.

    Parameters
    ----------
    video_dir : str
        The directory containing the videos.

    session_name : str
        The name of the session.

    Returns
    -------
    gimbal_positions : ndarray
        The gimbal positions for the session.
    """
    gimbal_file = join(video_dir, session_name + ".gimbal.npy")
    if not exists(gimbal_file):
        return None

    gimbal_positions = np.load(gimbal_file)
    gimbal_positions = median_filter(gimbal_positions, (5, 1, 1))
    gimbal_uvs = [
                mcc.project_points(gimbal_positions, ext, *intr)
                for ext, intr in zip(all_extrinsics, all_intrinsics)
            ]
    
    gimbal_uvs = {cam: uv for cam, uv in zip(camera_names, gimbal_uvs)}

    return gimbal_uvs


@click.command()
@click.argument("project_dir", type=str)
@click.argument("session_name", type=str)
@click.option("--show_steps", multiple=True, default=["HRNET", "TRIANGULATION", "GIMBAL"], help="Steps to include in the video. Pass this option multiple times to show multiple steps.")
@click.option("--output_dir", default="qc_videos", help="Directory to save the QC videos, relative to the project dir; or an absolute path.")
@click.option("--video_suffix", default="stages.mp4", help="Suffix of the qc video")
@click.option("--nframes", default=7200, help="Number of frames to use for the QC video (default = 60 sec * 120 fps = 7200 frames)")
@click.option("--fps", default=120, help="Fps of the video")
@click.option("--conf_threshold", default=0.25, help="Lower confidence bound to accept a keypoint")
@click.option("--overwrite", is_flag=True, help="Overwrite existing videos")
def main(
    project_dir,
    session_name,
    show_steps=["HRNET", "TRIANGULATION", "GIMBAL"], 
    output_dir="qc_videos",
    video_suffix="stages.mp4", 
    nframes=7200, 
    fps=120,
    conf_threshold=0.25,
    overwrite=False,
    ):
    """
    Make a quality control video for a session.

    Parameters
    ----------
    project_dir : str
        The path to the keypoints config project (config should be in this dir).

    session_name : str
        The name of the session to make the video for.
        (Key into the session_info dict of the config file.)

    show_steps : list of str, optional
        The steps to include in the video. Default is ["HRNET", "TRIANGULATION", "GIMBAL"].

    output_dir : str, optional
        The directory to save the video in. Default is "qc_videos".
        Can be a relative path to the project dir, or an absolute path.
        
    video_suffix : str, optional
        The suffix of the video file. Default is "stages.mp4".

    nframes : int, optional
        The number of frames to use for the video. Default is 7200.

    fps : int, optional
        The frames per second of the video. Default is 120.

    conf_threshold : float, optional
        The lower confidence bound to accept a keypoint. Default is 0.25.

    overwrite : bool, optional
        Overwrite existing videos. Default is False.
    """

    print(f"Making QC video for {session_name}")

    # Check if output dir is abspath, if not resolve it to one relative to the project dir
    if not os.path.isabs(output_dir):
        output_dir = join(project_dir, output_dir)

    # Check if output already exists, if not, skip
    output_path = join(output_dir, session_name + "." + video_suffix)
    if exists(output_path) and not overwrite:
        print(f"Output already exists for {session_name}, exiting")
        return

    # Deal with output setup
    if not exists(output_dir):
        os.mkdir(output_dir)

    # Get the set of videos for this session (ie top/bottom/sides/)
    config = load_config(project_dir)
    session_info = config["SESSION_INFO"][session_name]
    video_dir = session_info["video_dir"]
    video_names = session_info["videos"]
    
    # Projecting kps from triang/gimbal requires calibration files, check that we have that if needed
    require_calibn = any([any([base_step in step for base_step in ["TRIANGULATION", "GIMBAL"]]) for step in show_steps]) 
    if require_calibn and "calib_file" not in session_info:
        raise RuntimeError(f"No calibration file found for {session_name}, skipping")
    elif require_calibn:
        calib_path = session_info["calib_file"]
        all_extrinsics, all_intrinsics, camera_names = mcc.load_calibration(
            calib_path, "gimbal"
        )

        # Ensure video_names ordering matches camera_names order,
        # b/c camera_names order indicates the order in the calibration output, etc.
        video_names_dict = {
            v.split(".")[1]: v for v in video_names
        }  # ie "top": "20240101_J03001.top"
        video_names = [video_names_dict[c] for c in camera_names]
    elif not require_calibn:
        camera_names = [vid_name.split(".")[1] for vid_name in video_names]  # most complex video name could be /path/to/my/[recording_name]/[recording_name].[camera_name].[camera_serial].[first_frame_num].mp4, so we split on "." and take the second element to get the camera name

    # Dictionary to hold all the data
    all_uvs = {} # dictionary of uvs for each step, containing in turn a dictionary of uvs by video

    # Check if alignment of HRNet step is required due to dropped frames
    if any(["HRNET" in step for step in show_steps]) and session_info["alignment_required"]:
        alignment_file = join(session_info["video_dir"], "aligned_frame_numbers.csv")
        if not exists(alignment_file):
            print(f"Skipping session {session_name} because alignment file {alignment_file} not found")
            return
        align_df = pd.read_csv(alignment_file)  # cols are top, bottom, side1, ..., side4, trigger_number
    else:
        align_df = None

    ### Load the data for each step 
    for step in show_steps:

        # Figure out how to load data for this step
        if "HRNET" in step:
            output_name = config[step]["output_info"]["output_name"]  
            detection_uvs = load_hrnet_uvs(video_dir, video_names, output_name, conf_threshold, align_df=align_df)

            # If no HRNET files found, or for some reason lengths don't match (eg dropped frames), skip this step
            any_files = np.sum([uv is not None for uv in detection_uvs.values()]) > 0
            all_files_same_length = len(set([uv.shape[0] for uv in detection_uvs.values() if uv is not None])) == 1
            if any_files and all_files_same_length:
                all_uvs[step] = detection_uvs
            elif not any_files:
                print(f"No files found for {step}, skipping")
                all_uvs[step] = None
                continue
            elif not all_files_same_length:
                print(f"Files for {step} have different lengths, skipping.")
                print(f"Lengths: {({k: uv.shape[0] for k,uv in detection_uvs.items()})}")
                all_uvs[step] = None
                continue
        
            # If HRNET is the only step, set centroids to the mean of the detections
            if not any(["TRIANGULATION" in step for step in show_steps]):
                uvs_dims = list(all_uvs[step].values())
                uvs_dim = [uv.shape for uv in uvs_dims if uv is not None][0]
                centroids = np.round(gaussian_filter1d(np.nanmean([uv if uv is not None else np.zeros(uvs_dim) for uv in detection_uvs.values()], axis=2), 10, axis=1)).astype("int")

        elif "TRIANGULATION" in step:
            triang_uvs = load_triangulation_uvs(video_dir, session_name, all_extrinsics, all_intrinsics, camera_names)
            all_uvs[step] = triang_uvs

            # If triangulation is the last step, set centroids to the mean of the triangulations
            if "GIMBAL" not in show_steps:
                centroids = gaussian_filter1d(np.nanmean(triang_uvs, axis=2), 10, axis=1)
                
        # Load gimbal uvs
        elif "GIMBAL" in step:
            gimbal_uvs = load_gimbal_uvs(video_dir, session_name, all_extrinsics, all_intrinsics, camera_names)
            centroids = {cam: gaussian_filter1d(np.mean(uvs, axis=1), 10, axis=0) for cam, uvs in gimbal_uvs.items()}
            all_uvs[step] = gimbal_uvs

    # If no files found for any step, skip this session
    if np.sum([uv is not None for uv in all_uvs.values()]) == 0:
        print(f"No usable step data found for {session_name}, exiting")
        return

    # uvs_dims = list(all_uvs[show_steps[0]].values())
    # uvs_dim = [uv.shape for uv in uvs_dims if uv is not None][0]
    # all_uvs = np.stack([[uv if uv is not None else np.zeros(uvs_dim) for uv in step_uvs.values()] for step_uvs in all_uvs.values()])
    
    # Prepare to make the video
    readers = [OpenCVReader(join(video_dir, v + ".mp4")) for v in video_names]
    crop_size = 384
    print(f"Making video {output_path}...")
    with imageio.get_writer(
        output_path, pixelformat="yuv420p", fps=fps, quality=5
    ) as writer:
        
        # For each frame in the qc video
        for i in tqdm(range(nframes)):
            base_ims = [reader[i] for reader in readers]  # ncameras x h x w x 3
            frame = []

            # For each step, make a row of images, where each col is a camera view of kps for that step
            for step in show_steps:
                row = []

                # TODO: if all_uvs[step] is None, skip this step (just add a blank row?)

                # For each camera in this step
                for j, (base_im, camera_name) in enumerate(zip(base_ims, camera_names)):
                    vid_uvs = all_uvs[step][camera_name]  # nframes x nbodyparts x 2, or None if step not done for this video yet 

                    # Overlay keypoints if we have the data
                    if vid_uvs is not None:
                        frame_uvs = vid_uvs[i, ...]
                        im = overlay_keypoints_on_image(base_im.copy(), frame_uvs, edges)
                        im = crop_image(im, centroids[camera_name][i], crop_size)
                    
                    # Label the frame
                    if np.sum(im) == 0:
                        im = add_text(im, "CROP FAILED", (10, 180))
                    im = add_text(im, step, (10, 36))
                    im = add_text(im, camera_names[j], (10, 18))
                    row.append(im)
                    
                # Add the row to the frame
                frame.append(np.hstack(row))

            # Stack the rows to make the frame
            frame = np.vstack(frame)

            # Add the frame number
            frame = add_text(frame, repr(i), (10, frame.shape[0] - 12))
            frame = cv2.resize(frame, (256 * len(camera_names), 256 * len(show_steps)))

            # Write the frame to the video
            writer.append_data(frame)
            # print(f"Frame {i} done")
    
    # Report being done with the video
    print(f"Done with {session_name}")

    return


if __name__ == "__main__":
    main()
