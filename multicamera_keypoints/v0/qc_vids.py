# import gimbal
import os
from os.path import exists, join

import click
import cv2
import h5py
import imageio
from keypoint_moseq.util import get_edges
from keypoint_moseq.viz import overlay_keypoints_on_image, crop_image
import multicam_calibration as mcc
import networkx as nx
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm
from vidio.read import OpenCVReader


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


def load_config(project_dir, check_if_valid=True):
    """Load a project config file.

    Parameters
    ----------
    project_dir: str
        Directory containing the config file

    check_if_valid: bool, default=True
        Check if the config is valid using
        :py:func:`multicamera_keypoints.v0.io.check_config_validity`

    Returns
    -------
    config: dict
    """
    config_path = os.path.join(project_dir, "keypoint_config.yml")

    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    return config


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


use_bodyparts = bodyparts_hrnet_ordering[1:]
use_bodyparts_ix = np.array(
    [bodyparts_hrnet_ordering.index(bp) for bp in use_bodyparts]
)
edges = np.array(get_edges(use_bodyparts, skeleton))
node_order, parents = build_node_hierarchy(use_bodyparts, skeleton, "spine_low")
edges = np.argsort(node_order)[edges]

@click.command()
@click.argument("project_path")
@click.option("--show_steps", multiple=True, default=["HRNET", "TRIANGULATION", "GIMBAL"], help="Steps to include in the video. Pass this option multiple times to show multiple steps.")
@click.option("--video_suffix", default="stages.mp4", help="Suffix of the qc video")
@click.option("--nframes", default=7200, help="Number of frames to use for the QC video")
@click.option("--fps", default=120, help="Fps of the video")
@click.option("--conf_threshold", default=0.25, help="Lower confidence bound to accept a keypoint")
@click.option("--overwrite", is_flag=True, help="Overwrite existing videos")
def main(
    project_path, 
    show_steps=["HRNET", "TRIANGULATION", "GIMBAL"], 
    video_suffix="stages.mp4", 
    nframes=7200, 
    fps=120,
    conf_threshold=0.25,
    overwrite=False,
    ):
    
    config = load_config(project_path)


    # For each session analyzed, make one QC video
    for session, session_info in config["SESSION_INFO"].items():

        # # DEBUG
        # if session != "20240209_J03002": 
        #     continue

        print()
        print(session)

        # Get the set of videos for this session (ie top/bottom/sides/)
        video_dir = session_info["video_dir"]
        video_names = session_info["videos"]
        

        # Projecting kps from triang/gimbal requires calibration files, check that we have that if needed
        require_calibn = any([any([base_step in step for base_step in ["TRIANGULATION", "GIMBAL"]]) for step in show_steps]) 
        if require_calibn and "calib_file" not in session_info:
            print(f"No calibration file found for {session}, skipping")
            continue
        elif require_calibn:
            calib_path = session_info["calib_file"]
            all_extrinsics, all_intrinsics, camera_names = mcc.load_calibration(
                calib_path, "gimbal"
            )

            # Ensure video_names ordering matches camera_names order,
            # b/c camera_names order indicates the order in the calibration output, etc.
            video_names_dict = {
                v.split(".")[-1]: v for v in video_names
            }  # ie "top": "20240101_J03001.top"
            video_names = [video_names_dict[c] for c in camera_names]
        elif not require_calibn:
            camera_names = [vid_name.split(".")[1] for vid_name in video_names]  # most complex video name could be /path/to/my/[recording_name]/[recording_name].[camera_name].[camera_serial].[first_frame_num].mp4, so we split on "." and take the second element to get the camera name

        # Check if output already exists
        output_dir = join(project_path, "qc_videos")
        output_path = join(output_dir, session + "." + video_suffix)
        if exists(output_path) and not overwrite:
            print(f"Output already exists for {session}, skipping")
            continue

        # Deal with output setup
        if not exists(output_dir):
            os.mkdir(output_dir)

        # Dictionary to hold all the data
        all_uvs = {} # dictionary of uvs for each step, containing in turn a dictionary of uvs by video

        # Load the data for each step 
        print("Loading data")
        for step in show_steps:
            output_name = config[step]["output_info"]["output_name"]

            # Get HRNet data, if relevant
            if any(["HRNET" in step for step in show_steps]):
                detection_uvs = {}    

                # For each video, load the HRNET detections. If file not found, set to None.
                for i, v in enumerate(video_names):
                    detn_file = join(video_dir, v + "." + output_name)
                    if not exists(detn_file):
                        detection_uvs[v] = None
                        continue

                    # Read in the detections, masking out any below the conf threshold
                    with h5py.File(detn_file, "r") as h5:
                        uvs = h5["uv"][()][:, use_bodyparts_ix][:, node_order]
                        mask = h5["conf"][()][:, use_bodyparts_ix][:, node_order] < conf_threshold
                        uvs[mask] = np.nan
                        detection_uvs[v] = uvs

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

            # Get triangulation data, if relevant
            if any(["TRIANGULATION" in step for step in show_steps]):
                triang_uvs = {}
                triang_file = join(video_dir, session + ".robust_triangulation.npy")
                if not exists(triang_file):
                    all_uvs[step] = None
                else:

                    # Load the triangulation data
                    triang_positions = np.load(triang_file)
                    triang_positions = triang_positions[
                        :, use_bodyparts_ix][:, node_order]
                    
                    # Project the triangulated positions to each camera
                    triang_uvs = [
                        mcc.project_points(triang_positions, ext, *intr)
                        for ext, intr in zip(all_extrinsics, all_intrinsics)
                    ]
                    all_uvs[step] = triang_uvs

                # If triangulation is the last step, set centroids to the mean of the triangulations
                if "GIMBAL" not in show_steps:
                    centroids = gaussian_filter1d(np.nanmean(triang_uvs, axis=2), 10, axis=1)
                    
            # Load gimbal uvs
            # if any(["GIMBAL" in step for step in steps_to_use]):
            #     gimbal_positions = median_filter(
            #         np.load(join(video_dir, session + ".gimbal.npy")), (5, 1, 1)
            #     )
            #     gimbal_uvs = [
            #         mcc.project_points(gimbal_positions, ext, *intr)
            #         for ext, intr in zip(all_extrinsics, all_intrinsics)
            #     ]
            #     centroids = gaussian_filter1d(np.mean(gimbal_uvs, axis=2), 10, axis=1)
            #     all_uvs.append(gimbal_uvs)


        # If no files found for any step, skip this session
        if np.sum([uv is not None for uv in all_uvs.values()]) == 0:
            print(f"No usable step data found for {session}, skipping")
            continue

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
                    for j, base_im in enumerate(base_ims):
                        vid_uvs = all_uvs[step][video_names[j]]  # nframes x nbodyparts x 2, or None if step not done for this video yet 

                        # Overlay keypoints if we have the data
                        if vid_uvs is not None:
                            frame_uvs = vid_uvs[i, ...]
                            im = overlay_keypoints_on_image(base_im.copy(), frame_uvs, edges)
                            im = crop_image(im, centroids[j, i], crop_size)
                        
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
        print(f"Done with {session}")

    return


if __name__ == "__main__":
    main()
