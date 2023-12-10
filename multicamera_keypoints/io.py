import re
import numpy as np
import h5py
import joblib
import tqdm
import yaml
import os
from os.path import join, exists
import pandas as pd
from textwrap import fill
from pathlib import Path
import time

from multicamera_keypoints.file_utils import find_files_from_pattern
from multicamera_keypoints.vid_utils import count_frames

PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

config_comments = {
        "slurm_params": "parameters for submitting jobs to slurm",
        "weights_path": "path to pretrained weights",
        "output_name": "suffix for the outputted npy file, ie video_base_name.output_name.npy",
        "conda_env": "name of the conda environment to use for running the code",
        "verbose": "whether to print progress messages during fitting",
        "keypoint_colormap": "colormap used for visualization; see `matplotlib.cm.get_cmap` for options",
    }


def _build_yaml(sections, comments):
    text_blocks = []
    for title, data in sections:
        centered_title = f" {title} ".center(50, "=")
        text_blocks.append(f"\n\n{'#'}{centered_title}{'#'}")
        section_data = {title: data}  # Wrap the data under the title
        for key, value in section_data.items():
            text = yaml.dump({key: value}, default_flow_style=False).strip()
            # Add comments if they exist for the key
            if key in comments:
                text = f"\n{'#'} {comments[key]}\n{text}"
            text_blocks.append(text)
    return "\n".join(text_blocks)



def generate_config(project_dir, non_default_config=None, overwrite=False):
    """Generate a `config.yml` file with project settings. Default settings
    will be used unless overriden by a keyword argument.

    Parameters
    ----------
    project_dir: str
        A file `config.yml` will be generated in this directory.

    kwargs
        Custom project settings.
    """

    # Prevent overwriting existing config file
    config_path = join(project_dir, "config.yml")
    if exists(config_path) and not overwrite:
        raise ValueError(f"Config file already exists at {config_path}. Use `overwrite=True` to overwrite.")

    # Convert to absolute paths
    project_dir = os.path.abspath(project_dir)

    # Add info about the videos
    video_paths = find_files_from_pattern(project_dir, "*.avi", n_expected=6, error_behav="raise")
    vid_info = {}
    for video in video_paths:
        nframes = count_frames(video)
        camera = video.split(".")[-2]  # vid files are YYYYMMDD_HHMMSS.camera.avi
        vid_info[camera] = {
            "video_path": video,
            "nframes": nframes
            }

    # Generate info for a dummy processing step
    dummy_sec_per_frame = 0.001
    dummy = {
        "slurm_params": {
            "mem": "4GB",
            "gpu": False,
            "sec_per_frame": dummy_sec_per_frame,  
            "ncpus": 1,
        },
        "wrap_params":{
            "func_path": join(PACKAGE_DIR, "v0", "dummy.py"),
            "conda_env": "dataPy_NWB2"
        },
        "func_args": {
            "video_path": "{video_path}",
        },
        "output_info": {
            "output_name": "dummy",
        },
    }

    # Generate info for the centernet
    centernet_sec_per_frame = 0.021
    centernet = {
        "slurm_params": {
            "mem": "4GB",
            "gpu": True,
            "sec_per_frame": centernet_sec_per_frame,  # 75 min/job x 60 / (30*60*120 frames/job) = 0.021 sec/frame
            "ncpus": 2,
        },
        "wrap_params":{
            "func_path": join(PACKAGE_DIR, "v0", "segmentation.py"),
            "conda_env": "dataPy_torch"  # TODO: make this dynamic?
        },
        "func_args": {
            "video_path": "{video_path}",
            "weights_path": "/n/groups/datta/Jonah/kpms_reviews_6cam_thermistor/20230928_avi_compression/train_hrnet-v2/weights/centernet.pth",  # TODO: make this dynamic?
        },
        "output_info": {
            "output_name": "centroid",
        },
    }

    # hrnet params

    other = {
        "video_dir": project_dir,  # assume config is in the dir with videos by default
        "verbose": False,
        "keypoint_colormap": "autumn",
    }

    sections = [
        ("VID_INFO", vid_info),
        ("CENTERNET", centernet),
        ("DUMMY", dummy),
        ("OTHER", other),
    ]

    with open(config_path, "w") as f:
        f.write(_build_yaml(sections, config_comments))

    # Update the config with any non-default settings
    config = load_config(project_dir)
    if non_default_config is not None:
        update_config(config, non_default_config)

    return


def check_config_validity(config):

    if isinstance(config, str):
        config = load_config(config)

    error_messages = []

    # Check that required paths exist
    top_levels = ["CENTERNET", "DUMMY"]
    for top_level in top_levels:

        if "weights_path" in config[top_level]["func_args"]:
            check = config[top_level]["func_args"]["weights_path"]
            if not os.path.exists(check):
                error_messages.append(
                        f"Path to weights does not exist: {check}"
                    )

        check = config[top_level]["wrap_params"]["func_path"]
        if not os.path.exists(check):
            error_messages.append(
                    f"Path to weights does not exist: {check}"
                )


    # Check that conda envs exist
    if not exists(join(os.environ["CONDA_PREFIX"], "envs", config['CENTERNET']["wrap_params"]["conda_env"])):
        error_messages.append(
                f"Conda env does not exist: {config['CENTERNET']['wrap_params']['conda_env']}"
            )

    # Check that all videos have same length
    nframes = {k: v["nframes"] for k,v in config["VID_INFO"].items()}
    if len(set([nf for nf in nframes.values()])) > 1:
        error_messages.append(
                f"Videos have different lengths: {nframes}"
            )


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
    config_path = os.path.join(project_dir, "config.yml")

    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    if check_if_valid:
        check_config_validity(config)

    return config


def update_config(project_dir, new_config, verbose=True):
    """Update the config file stored at `project_dir/config.yml`.
    """
    def recursive_update(config, updates):
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict):
                # If the key is a dictionary, recurse
                recursive_update(config[key], value)
            else:
                # Otherwise, update the value directly
                if verbose:
                    print(f'updating {key} to {value}')
                config[key] = value
        return config

    config = load_config(
        project_dir, check_if_valid=False
    )
    config = recursive_update(config, new_config)
    sections = [(k, v) for k,v in config.items()]
    with open(os.path.join(project_dir, "config.yml"), "w") as f:
        f.write(_build_yaml(sections, config_comments))
    time.sleep(0.1)  # wait for file to be written