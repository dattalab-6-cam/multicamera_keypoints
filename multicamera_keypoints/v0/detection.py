import os
from os.path import join, exists

import click
import cv2
import h5py
import numpy as np
import tqdm
from o2_utils.selectors import find_files_from_pattern
import pandas as pd
from scipy.ndimage import median_filter
from vidio.read import OpenCVReader

from multicamera_keypoints.io import load_config

# see also imports under main()
from multicamera_keypoints.v0.hrnet import HRNet
from multicamera_keypoints.vid_utils import crop_image

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


def colormap_with_distinct_lowend(base_cmap="viridis", low_end=0.25):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    # Define the color for the 0 to 0.25 range (dark red to light red)
    red_colors = [(0.5, 0, 0), (1, 0.5, 0.5)]  # Dark Red to Light Red
    red_map = LinearSegmentedColormap.from_list("red_grad", red_colors, N=256)

    # Get the viridis colormap for the 0.25 to 1 range
    viridis = plt.cm.get_cmap('viridis', 256)

    # Create a new colormap by combining the two
    boundary = np.floor(256 * low_end).astype(int)

    # First 64 entries (0.25 of 256) are red gradient, the rest are viridis
    new_colors = np.vstack((red_map(np.linspace(0, 1, boundary)), viridis(np.linspace(0, 1, 256 - boundary))))
    custom_colormap = ListedColormap(new_colors, name='RedViridis')

    return custom_colormap


def make_kp_detection_quality_plots(project_dir, processing_step="HRNET", relative_save_dir="./validations", style_sheet=None, conf_threshold=0.25, overwrite=False):
    """ Assess keypoint detection quality with a variety of metrics.

    Apart from showing the distribution of confidences, only considers detections with confidence above conf_threshold.

    -- Distribution of detection confidences per camera / type of bodypart
    -- Heatmap of detection confidences over time (plus max confidences across all cameras per bodypart)
    -- Distribution of frame-to-frame displacements for each keypoint
    -- Inter-keypoint distances for low-variance pairs like front/back of hindpaws (in UV space, so not perfect)

    Parameters
    ----------
    project_dir : str
        The directory containing the project data.

    processing_step : str, optional
        The processing step to assess. The default is "HRNET". Only HRNET-based steps are supported. For example, if you 
        run HRNET with a different finetuned network / algorithm and specificy a distinct processing step (eg "HRNET.v2"),
        you can assess the quality of that specific step.

    relative_save_dir : str, optional
        The directory to save the validation plots, appended to [project_dir]/keypoint_bach/[processing_step]. The default is "./validations".

    style_sheet : str, optional
        The matplotlib style sheet to use for the plots. The default is None.

    conf_threshold : float, optional
        The confidence threshold to use for filtering out low-confidence detections. The default is 0.25.

    overwrite : bool, optional
        Whether to overwrite existing output files. The default is False.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    if style_sheet is not None:
        plt.style.use(style_sheet)

    if "HRNET" not in processing_step:
        raise ValueError(f"Processing step {processing_step} is not HRNET-based, cannot make keypoint quality plots")
    
    config = load_config(project_dir)
    output_name = config[processing_step]["output_info"]["output_name"]

    # Sets of keypoints to assess together for confidences
    paw_keypoint_idx = [bodyparts_hrnet_ordering.index(bp) for bp in ["left_fore_paw", "right_fore_paw", "left_hind_paw_front", "right_hind_paw_front", "left_hind_paw_back", "right_hind_paw_back"]]
    head_keypoint_idx = [bodyparts_hrnet_ordering.index(bp) for bp in ["left_ear", "right_ear", "forehead", "nose_tip"]]
    spine_keypoint_idx = [bodyparts_hrnet_ordering.index(bp) for bp in ["tail_tip", "tail_base", "spine_low", "spine_mid", "spine_high"]]

    # Sets of theoretically low-variance keypoint distances (this is just in UV space for now, not in mm)
    low_var_distance_pairs = {
        "bottom": [
            ("left_hind_paw_front", "left_hind_paw_back"),
            ("right_hind_paw_front", "right_hind_paw_back"),
        ],
        "top": [
            ("nose_tip", "forehead"),
            ("left_ear", "right_ear"),
        ]
    }
    
    for session, session_info in config["SESSION_INFO"].items():

        # Prep some vars
        video_dir = session_info["video_dir"]
        video_names = session_info["videos"]
        detection_uvs = {}
        all_camera_confs = []

        # Check if the output files already exist
        # (Just uses one figure as a proxy for them all, for now)
        output_dir = join(project_dir, "keypoint_batch", processing_step, relative_save_dir)
        if os.path.exists(join(output_dir, f"{session}_kp_confidences.png")) and not overwrite:
            print(f"Skipping session {session} because output files already exist")
            continue

        # Check that there are enough detection files to proceed
        detn_files = [join(video_dir, v + "." + output_name) for v in video_names]
        n_existing_detn_files = sum([os.path.exists(f) for f in detn_files])
        if n_existing_detn_files <= 1:
            print(f"Skipping session {session} because only {n_existing_detn_files} detection files found")
            continue

        # Check if alignment is required, if so look for alignment file in the video dir
        if session_info["alignment_required"]:
            alignment_file = join(session_info["video_dir"], "aligned_frame_numbers.csv")
            if not exists(alignment_file):
                print(f"Skipping session {session} because alignment file {alignment_file} not found")
                continue
            align_df = pd.read_csv(alignment_file)  # cols are top, bottom, side1, ..., side4, trigger_number
            max_n_frames = align_df.shape[0]
        else:
            vid = session_info["videos"][0]
            max_n_frames = config["VID_INFO"][vid]["nframes"]
            align_df = pd.DataFrame(
                {(video.split(".")[1]): np.arange(max_n_frames) for video in session_info["videos"]}
            )

        # Prepare figures
        conf_fig, conf_axs = plt.subplots(1, 3, figsize=(9,3))  # will show histograms of kp confidences, for paws, head, and spine kps
        conf_heatmap_fig, conf_heatmap_axs = plt.subplots(3, 3, figsize=(12,9))  # will show heatmap of confidences over time
        conf_heatmap_axs = conf_heatmap_axs.ravel()
        dist_fig, dist_axs = plt.subplots(1, len(low_var_distance_pairs), figsize=(6,3))  # will show histograms of distances between low-variance pairs of kps
        dist_axs = {k: ax for k, ax in zip(low_var_distance_pairs.keys(), dist_axs)}
        temporal_fig, temporal_axs = plt.subplots(4, 4, figsize=(12,12))  # will show histograms of frame-to-frame displacement for each kp
        temporal_axs = temporal_axs.ravel()

        # Loop over the videos in this session
        for i, v in enumerate(video_names):
            camera = v.split(".")[1]
            
            # Load the detections and their confidences
            detn_file = detn_files[i]
            if not os.path.exists(detn_file):
                detection_uvs[v] = None
                continue
            array_dict = load_arrays_from_h5(detn_file)
            raw_uvs = array_dict["uv"]  # shape (n_frames, n_kps, 2)
            confs = array_dict["conf"] # shape (n_frames, n_kps)

            # Align the detections to the video frames
            aligned_confs = np.nan * np.zeros((max_n_frames, confs.shape[1]))
            align_vec = align_df[camera].values
            aligned_confs[~pd.isnull(align_vec), :] = confs
            all_camera_confs.append(aligned_confs)

            # Set low-conf detections to nan
            cleaned_uvs = raw_uvs.copy()
            cleaned_uvs[confs < conf_threshold] = np.nan
            
            # Extract the confidences for the keypoint sets
            paw_confs = confs[:, paw_keypoint_idx]
            head_confs = confs[:, head_keypoint_idx]
            spine_confs = confs[:, spine_keypoint_idx]

            # Plot hists of detection confidences
            conf_axs[0].hist(paw_confs.flatten(), bins=np.arange(0, 1, 0.01), histtype="step", label=v.split(".")[-1], density=True, zorder=-i)
            conf_axs[1].hist(head_confs.flatten(), bins=np.arange(0, 1, 0.01), histtype="step", label=v.split(".")[-1], density=True, zorder=-i)
            conf_axs[2].hist(spine_confs.flatten(), bins=np.arange(0, 1, 0.01), histtype="step", label=v.split(".")[-1], density=True, zorder=-i)

            # Plot the heatmap of confidences
            im = conf_heatmap_axs[i].imshow(confs.T, aspect="auto", cmap=colormap_with_distinct_lowend(), vmin=0, vmax=1, interpolation="none",)
            cb = plt.colorbar(im)
            cb.set_label("Confidence")
            conf_heatmap_axs[i].set(
                xlabel="Frame", 
                ylabel="Keypoint", 
                title=f"Confs. for {camera}",
                xticks=np.arange(0, len(confs), 50000),
                xticklabels=np.arange(0, len(confs), 50000),
                yticks=np.arange(0, len(bodyparts_hrnet_ordering), 1),
                yticklabels=bodyparts_hrnet_ordering
            )

            # Distances between low-variance pairs of keypoints
            if camera in low_var_distance_pairs:
                for iPair, (bp1, bp2) in enumerate(low_var_distance_pairs[camera]):
                    bp1_idx = bodyparts_hrnet_ordering.index(bp1)
                    bp2_idx = bodyparts_hrnet_ordering.index(bp2)
                    dists = np.linalg.norm(raw_uvs[:, bp1_idx] - raw_uvs[:, bp2_idx], axis=-1)
                    high_conf_dists = dists[(confs[:, bp1_idx] > conf_threshold) & (confs[:, bp2_idx] > conf_threshold)]
                    low_conf_dists = dists[(confs[:, bp1_idx] <= conf_threshold) | (confs[:, bp2_idx] <= conf_threshold)]
                    dist_axs[camera].hist(high_conf_dists, bins=np.arange(0, 200, 5), color=f"C{iPair}", histtype="step", label=f"{bp1} -->\n {bp2}", zorder=-i)
                    dist_axs[camera].hist(low_conf_dists, bins=np.arange(0, 200, 5), color=f"C{iPair}", histtype="step", label="(low conf)", zorder=-i, linestyle="--", lw=0.25)
                    dist_axs[camera].legend(ncol=2, fontsize=4)

            # Temporal displacements of each keypoint
            for iKp, kp in enumerate(bodyparts_hrnet_ordering):
                uv = cleaned_uvs[:, iKp, :]
                displacements = np.linalg.norm(np.diff(uv, axis=0), axis=-1)
                temporal_axs[iKp].hist(displacements, bins=np.concatenate([np.arange(0, 50, 2), [np.inf]]), histtype="step", label=v.split(".")[-1], zorder=-i)
                temporal_axs[iKp].set(
                    title=kp,
                    xlabel="Displacement (pixels)",
                    ylabel="Occurrences",
                    yscale="log",
                )
                temporal_axs[iKp].legend(ncol=2, fontsize=4)
            
        # Plot the max confidence for each keypoint across all cams, for each frame
        max_confs = np.nanmax(all_camera_confs, axis=0)
        conf_heatmap_axs[-1].set_axis_off()
        conf_heatmap_axs[-3].set_axis_off()
        conf_heatmap_axs[-2].imshow(max_confs.T, aspect="auto", cmap=colormap_with_distinct_lowend(), vmin=0, vmax=1, interpolation="none",)
        cb = plt.colorbar(im)
        cb.set_label("Max Confidence")
        conf_heatmap_axs[-2].set(
            xlabel="Frame", 
            ylabel="Keypoint", 
            title="Max Confs. across all cameras",
            xticks=np.arange(0, len(confs), 50000),
            xticklabels=np.arange(0, len(confs), 50000),
            yticks=np.arange(0, len(bodyparts_hrnet_ordering), 1),
            yticklabels=bodyparts_hrnet_ordering
        )


        # Format the plots
        for ax, name in zip(conf_axs, ["Paw", "Head", "Spine"]):
            ax.set(
                xlabel="Keypoints' confidences",
                ylabel="Occurrences",
                title=f"{name} confidences"
            )
            ax.legend(ncol=2, fontsize=6)
        conf_fig.suptitle(f"Confidence distributions for {session}")
        conf_fig.tight_layout()

        conf_heatmap_fig.tight_layout(h_pad=0.1, w_pad=0.1)

        for camera, ax  in dist_axs.items():
            ax.set(
                xlabel="Distance (pixels)",
                ylabel="Occurrences",
                title=f"Distances from {camera} camera viewpoint"
            )
        dist_fig.suptitle(f"Distances for {session}")
        dist_fig.tight_layout()

        temporal_fig.suptitle(f"Temporal displacements for {session}")
        temporal_fig.tight_layout()

        # Save the plots
        os.makedirs(output_dir, exist_ok=True)
        conf_fig.savefig(join(output_dir, f"{session}_kp_confidences.png"))
        dist_fig.savefig(join(output_dir, f"{session}_low_var_distances.png"))
        conf_heatmap_fig.savefig(join(output_dir, f"{session}_confidences_heatmap.png"))
        temporal_fig.savefig(join(output_dir, f"{session}_temporal_displacements.png"))

        # Close the plots
        plt.close("all")

    return

def make_config(
    PACKAGE_DIR,
    weights_path,
    sec_per_frame=0.087,
    output_name_suffix=None,
    step_dependencies=None,
):
    """Create a default config for the HRNET step.

    Parameters
    ----------
    PACKAGE_DIR : str
        The directory where the package is installed.

    weights_path : str or dict
        If string: the path to a directory containing weights for the hrnet model, with weights named "hrnet_bottom.pth", "hrnet_top.pth", and "hrnet_side.pth".
        If dict: a dictionary with keys "bottom", "top", and "side", each containing the path to the corresponding weights file.
        
    sec_per_frame : float, optional
        The number of seconds per frame for the hrnet step. The default is 0.087.
        
    output_name_suffix : str, optional
        The suffix to add to the output name. The default is None.
        Example: "v2" --> "keypoints.v2.h5", and the step name 
        will be "HRNET.v2".

    step_dependencies : list, optional
        The list of step names for the dependencies of this step. The default is ["CENTERNET"].
        These steps will be checked for completion before running this step.

    Returns
    -------
    hrnet_config : dict
        The configuration for the hrnet step. 

    step_name : str
        The name of the detection step. (default: "HRNET")
    """

    # Set the output name and step name
    if output_name_suffix is not None:
        output_name = f"keypoints.{output_name_suffix}.h5"
        step_name = f"HRNET.{output_name_suffix}"
    else:
        output_name = "keypoints.h5"
        step_name = "HRNET"

    # Set the step_dependencies, if not given otherwise.
    if step_dependencies is None:
        step_dependencies = ["CENTERNET"]

    # Check that the weights_path is valid
    if isinstance(weights_path, str):
        assert os.path.isdir(weights_path), f"weights_path {weights_path} is not a directory"
    elif isinstance(weights_path, dict):
        for key, val in weights_path.items():
            assert os.path.isfile(val), f"weights_path[{key}] {val} is not a file"
        assert set(weights_path.keys()) == {"bottom", "top", "side"}, f'weights_path keys {weights_path.keys()} must be "bottom", "top", and "side"'

    # Create the hrnet_config dictionary
    hrnet_config = {
        "slurm_params": {
            "mem": "4GB",
            "gpu": True,
            "sec_per_frame": sec_per_frame,  # 75 min/job x 60 / (30*60*120 frames/job) = 0.021 sec/frame
            "ncpus": 2,
            "jobs_in_progress": {},
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "detection.py"),
            "conda_env": "dataPy_torch2",  # TODO: make this dynamic?
        },
        "func_args": {  # NB: these args **must** be in the right order here.
            "video_path": "{video_path}",
            "weights_path": weights_path,
        },
        "output_info": {
            "output_name": output_name,
        },
        "step_dependencies": step_dependencies,
        "pipeline_info": {
            "processing_level": "video",
        },
    }

    return hrnet_config, step_name



def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def parse_heatmap(heatmap, downsample=4):
    B, C, H, W = heatmap.shape
    flat_heatmap = heatmap.reshape((B, C, -1))
    maxima = torch.argmax(flat_heatmap, dim=-1)

    u = maxima % W
    v = torch.div(maxima, W, rounding_mode="floor")
    uv = downsample * torch.stack((u, v), dim=-1)

    confidence = torch.gather(flat_heatmap, -1, maxima[..., None])[..., 0]
    confidence = torch.clip(confidence, 0, 1)

    uv = to_numpy(uv).astype(np.int32)
    confidence = to_numpy(confidence).astype(np.float32)
    return uv, confidence


def load_model(weights_filepath, use_cpu=False):
    if use_cpu:
        state_dict = torch.load(weights_filepath, map_location=torch.device("cpu"))
    else:
        state_dict = torch.load(weights_filepath)
    nof_joints = state_dict["final_layer.weight"].shape[0]
    model = HRNet(nof_joints=nof_joints)
    model.load_state_dict(state_dict)
    return model, nof_joints


def apply_model_to_image(model, im, centroid, clahe, use_cpu=False):
    with torch.no_grad():
        im = crop_image(im, centroid, 512)
        im = clahe.apply(im[:, :, 0])
        x = im[None, None].astype(np.float32) / 255
        if use_cpu:
            y_pred = model(torch.Tensor(x))
        else:
            y_pred = model(torch.Tensor(x).to("cuda"))
        uv, conf = parse_heatmap(y_pred, downsample=2)
    uv = uv[0] + centroid[None, None] - 256

    return uv, conf[0]


def save_arrays_as_h5(save_path, array_dict):
    with h5py.File(save_path, "w") as h5f:
        for key, arr in array_dict.items():
            h5f.create_dataset(key, shape=arr.shape)
            h5f[key][:] = arr  # if you just write h5f[key] = arr, it will try to create the dataset and fail b/c it already exists.
    return


def load_arrays_from_h5(save_path):
    with h5py.File(save_path, "r") as h5f:
        array_dict = {key: h5f[key][:] for key in h5f.keys()}
    return array_dict


@click.command()
@click.argument("vid_path")
@click.argument("weights_path")
@click.option("--output_name", default="keypoints.h5", help="name of output file")
@click.option("--save_every", default=1000, help="frequency to save checkpoints")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output")
@click.option("--ignore_checkpoints", is_flag=True, help="Ignore existing checkpoints")
@click.option("--use_cpu", is_flag=True, help="Force to run on a CPU instead of a gpu (very slow -- use for debugging only)")
def main(vid_path, weights_path, output_name="keypoints.h5", save_every=1000, overwrite=False, ignore_checkpoints=False, use_cpu=False):
    """ Detects keypoints on a video using HRNet.

    This 

    Parameters
    ----------
    vid_path : str
        The path to the video file to process. Should contain the string either "bottom", "top", or "side" to indicate 
        which HRNet to use to detect the keypoints.

    weights_path : str or dict
        If string: the path to a directory containing weights for the hrnet model, with weights named "hrnet_bottom.pth", "hrnet_top.pth", and "hrnet_side.pth".
        If dict: a dictionary with keys "bottom", "top", and "side", each containing the path to the corresponding weights file.

    output_name : str, optional
        The suffix to use for the saved keypoint detections. The default is "keypoints.h5",
        and the final default filename is:
            os.path.splitext(vid_path) + "." + output_name

    save_every : int, optional
        The frequency to save checkpoints. The default is every 1000 frames.

    overwrite : bool, optional
        Whether to overwrite existing output file. The default is False.
        If False and the final output file exists, the program will exit.

    ignore_checkpoints : bool, optional
        Whether to ignore existing checkpoints. The default is False.
        If true, will re-start the detection process even if a checkpoint file exists.
        Otherwise, will attempt to start from the next frame after the checkpoint.

    Returns
    -------
    None

    """

    # Validate the weights paths
    if isinstance(weights_path, str):
        assert os.path.isdir(weights_path), f"weights_path {weights_path} is not a directory"
    elif isinstance(weights_path, dict):
        for key, val in weights_path.items():
            assert os.path.isfile(val), f"weights_path[{key}] {val} is not a file"
        assert set(weights_path.keys()) == {"bottom", "top", "side"}, f'weights_path keys {weights_path.keys()} must be "bottom", "top", and "side"'

    # Validate the output name
    assert output_name.split(".")[-1] == "h5", "Output name for keypoint detection must end in .h5"

    # Find the weights file for this particular video.
    camera = vid_path.split("/")[-1].split(".")[1]
    if isinstance(weights_path, str):
        if "bottom" in camera:
            weights_file = find_files_from_pattern(weights_path, "hrnet_bottom.pth")
        elif "top" in camera:
            weights_file = find_files_from_pattern(weights_path, "hrnet_top.pth")
        elif "side" in camera:
            weights_file = find_files_from_pattern(weights_path, "hrnet_side.pth")
        else:
            raise ValueError(f"Unexpected camera name detected from video: {camera}")
    elif isinstance(weights_path, dict):
        if "bottom" in camera:
            weights_file = weights_path["bottom"]
        elif "top" in camera:
            weights_file = weights_path["top"]
        elif "side" in camera:
            weights_file = weights_path["side"]
        else:
            raise ValueError(f"Unexpected camera name detected from video: {camera}")

    # Begin the detection process
    print(f"Detecting keypoints on {vid_path} using weights from {weights_file}...")

    # Set the output name and check for any overwriting issues
    video_name, _ = os.path.splitext(vid_path)
    save_path = video_name + "." + output_name
    if os.path.exists(save_path) and not overwrite:
        print(f'Output file {save_path} exists, exiting!')
        return
    else:
        print(f"Results will be saved to {save_path}")

    # Load the model
    model, nof_joints = load_model(weights_file, use_cpu=use_cpu)
    if use_cpu:
        model = model.eval()
    else:
        model = model.eval().to("cuda")
    

    # Load centroid info and a CLAHE normalizer for the video
    centroids = median_filter(np.load(video_name + ".centroid.npy"), (11, 1)).astype(int)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))

    # Open the video
    reader = OpenCVReader(vid_path)
    nframes = len(reader)

    # Check for any checkpoint files
    checkpoint_file = save_path.replace(".h5", ".CHECKPOINT.h5")
    if os.path.exists(checkpoint_file) and not ignore_checkpoints:
        print(f"Found checkpoint file {checkpoint_file}, loading and continuing...")
        array_dict = load_arrays_from_h5(checkpoint_file)
        all_uvs = array_dict["uv"]
        all_confs = array_dict["conf"]
        # with h5py.File(checkpoint_file, "r") as h5f:
        #     all_uvs = h5f["uv"][:]
        #     all_confs = h5f["conf"][:]
        start_frame = np.where(np.sum(all_uvs, axis=(1, 2)) == 0)[0][0]
        print(f"Starting from frame {start_frame}...")
    elif os.path.exists(save_path) and ignore_checkpoints:
        print("Checkpoint file exists but ignoring...")
    else:
        print("No checkpoint found, beginning detection from frame 0...")
        all_uvs = np.zeros((nframes, nof_joints, 2))
        all_confs = np.zeros((nframes, nof_joints))
        start_frame = 0
    
    # Begin the detection process
    for i in tqdm.trange(start_frame, nframes):
        im = reader[i]
        uv, conf = apply_model_to_image(model, im, centroids[i], clahe, use_cpu=use_cpu)
        all_uvs[i, ...] = uv
        all_confs[i, :] = conf
        if i % save_every == 0:
            array_dict = dict(uv=all_uvs, conf=all_confs)
            save_arrays_as_h5(checkpoint_file, array_dict)
            print(f"Saved checkpoint at frame {i}...")
    
    # Save the final results
    array_dict = dict(uv=all_uvs, conf=all_confs)
    save_arrays_as_h5(save_path, array_dict)

    # Remove the now-redundant checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("Done!")


if __name__ == "__main__":
    import torch  # protect this import from __init__ so that we can use the rest of the module on a CPU
    main()
