import os
from os.path import exists, join
from pathlib import Path
from textwrap import fill

import yaml
from o2_utils.selectors import find_files_from_pattern

# from multicamera_keypoints.file_utils import find_files_from_pattern
from multicamera_keypoints.vid_utils import count_frames_cached

PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSING_STEPS = ["CENTERNET", "HRNET", "TRIANGULATION", "GIMBAL"]  # not including calibration and whatnot

config_comments = {
    "slurm_params": "parameters for submitting jobs to slurm",
    "weights_path": "path to pretrained weights",
    "output_name": "suffix for the outputted npy file, ie video_base_name.output_name.npy",
    "conda_env": "name of the conda environment to use for running the code",
    "verbose": "whether to print progress messages during fitting",
    "keypoint_colormap": "colormap used for visualization; see `matplotlib.cm.get_cmap` for options",
}

config_name = "keypoint_config.yml"


def _build_yaml(sections, comments):
    """Build a yaml file from a list of sections and comments.

    Structure of the yaml file will look like this:
    ```
    #==================== SECTION_TITLE ====================#
    SECTION_TITLE:
        key: value
        subsection_key:
            # Comment
            nested_key: nested_value
    ```

    Parameters
    ----------
    sections : list of tuples
        Each tuple is a section of the yaml file. The first element is the
        section title, and the second element is a dictionary of key-value
        pairs.

    comments : dict
        A dictionary of comments for each key in the yaml file.

    Returns
    -------
    text : str
        The yaml file as a string.
    """
    text_blocks = []
    for title, data in sections:
        centered_title = f" {title} ".center(50, "=")
        text_blocks.append(f"\n\n{'#'}{centered_title}{'#'}")
        section_data = {title: data}  # Wrap the data under the title
        for key, value in section_data.items():
            text = yaml.dump({key: value}, default_flow_style=False, sort_keys=False).strip()
            # Add comments if they exist for the key
            if key in comments:
                text = f"\n{'#'} {comments[key]}\n{text}"
            text_blocks.append(text)
    return "\n".join(text_blocks)


def add_videos_to_config(
    project_dir, video_paths, overwrite=False
):
    
    # Find the existing videos
    config = load_config(project_dir)
    existing_vid_info = config["VID_INFO"]

    # Remove videos that are already in the config if not overwriting
    if not overwrite:
        video_paths = {v for v in video_paths if os.path.basename(v.split(".mp4")[0]) not in existing_vid_info}

    if len(video_paths) == 0:
        print("No new videos to add to config")
        return

    # Get info about the new videos (just frame num and path for now)
    new_vid_info = _get_video_info(video_paths)

    # Add key/val pairs specific to each processing step
    for vid_info in new_vid_info.values():
        vid_info.update({f"{step}_done": False for step in PROCESSING_STEPS})
    
    # Update the config
    existing_vid_info.update(new_vid_info)
    update_config(project_dir, {"VID_INFO": existing_vid_info}, verbose=False)

    # Report to user 
    print(f"Added {len(new_vid_info)} video(s) to config")

    return


def add_calibrations_to_config(
    project_dir, calibration_paths, board_size, square_size, overwrite=False
):
    
    # Find the existing calibrations
    config = load_config(project_dir)
    existing_calib_info = config.get("CALIBRATION_VIDEOS", {})
    print(existing_calib_info)

    # Remove calibrations that are already in the config if not overwriting
    if not overwrite:
        print(os.path.basename(calibration_paths[0].split(".mp4")[0]))
        calibration_paths = {v for v in calibration_paths if os.path.basename(v.split(".mp4")[0]) not in existing_calib_info}

    # Get info about the new calibrations
    _vid_info = _get_video_info([find_files_from_pattern(p, "*.top.mp4", n_expected=1, error_behav="raise") for p in calibration_paths])
    calib_info = {}
    for (calib_name, vid_info), calib_path in zip(_vid_info.items(), calibration_paths):
        short_name = calib_name.split(".")[0]
        calib_info[short_name] = {}
        calib_info[short_name]["video_dir"] = calib_path
        calib_info[short_name]["nframes"] = vid_info["nframes"]

    # Add key/val pairs specific to the calibration
    for vid_info in calib_info.values():
        vid_info.update(dict(CALIBRATION_done=False, ok=False, board_size=board_size, square_size=square_size))
    
    existing_calib_info.update(calib_info)
    update_config(project_dir, {"CALIBRATION_VIDEOS": existing_calib_info}, verbose=False)
    print(f"Added {len(calib_info)} calibration(s) to config")
    return

def _get_video_info(video_paths):
    """

    Returns
    -------
    vid_info: dict
        Keys are session names, values are dictionaries with keys "video_path" and "nframes".
    """
    # Add info about the videos
    # video_paths = find_files_from_pattern(project_dir, "*.avi", n_expected=6, error_behav="raise")
    vid_info = {}
    for video in video_paths:
        try:
            print(video)
            nframes = count_frames_cached(video)
        except Exception as e:
            print(f"Error counting frames in {video}, skipping...")
            print(f"Error: {e}")
            continue
        video_name, _ = os.path.splitext(os.path.basename(video))
        session_name = video_name.split(".")[0]
        vid_info[video_name] = {"video_path": os.path.abspath(video), "nframes": nframes, "session_name": session_name}
    return vid_info


def generate_config(
    project_dir, video_paths, weights_path, gimbal_params_path, non_default_config=None, overwrite=False
):
    """Generate a `config.yml` file with project settings. Default settings
    will be used unless overriden by a keyword argument.

    Assumes your videos are organized such that the video prefix (before the first ".") matches the session name.

    ```
    project_dir/
        session1/
            session1.top.mp4
            session1.side1.mp4
            ...
        session2/  
            session2.top.mp4
            session2.side1.mp4
            ...
    ```

    Parameters
    ----------
    project_dir: str
        A file `config.yml` will be generated in this directory.

    video_paths: list of str
        A list of video paths.

    weights_path: str
        Path to the pretrained weights. Files in this folder should be centernet.pth, hrnet_side.pth, hrnet_top.pth, and hrnet_bottom.pth.

    gimbal_params_path: str
        Path to the gimbal params file.

    non_default_config: dict, optional
        A dictionary of non-default settings. See below for details.

    overwrite: bool, optional
        Whether to overwrite an existing config file.

    Returns
    -------
    None
    """

    # import pdb

    # Prevent overwriting existing config file
    config_path = join(project_dir, config_name)
    if exists(config_path) and not overwrite:
        raise ValueError(
            f"Config file already exists at {config_path}. Use `overwrite=True` to overwrite."
        )

    # Get video info
    all_vid_info = _get_video_info(video_paths)

    # Add key/val pairs specific to each processing step
    for vid_info in all_vid_info.values():
        vid_info.update({"CENTERNET_done": False, "HRNET_done": False})

    # Parse videos into sessions
    sessions = [v["session_name"] for v in all_vid_info.values()]
    session_dirs = [os.path.dirname(v["video_path"]) for v in all_vid_info.values()]
    uq_pairings = list(set(list(zip(sessions, session_dirs))))
    all_session_info = {}
    for session, _dir in uq_pairings:
        all_session_info[session] = {
            "videos": [], 
            "CALIBRATION_done": False,
            "TRIANGULATION_done": False,
            "GIMBAL_done": False,
            "video_dir": _dir,
            "ready_for_processing": False,  # whether all videos are done with centernet and hrnet
        }
    
    for vid_name in all_vid_info:
        s = all_vid_info[vid_name]["session_name"]
        all_session_info[s]["videos"].append(vid_name)
    
    # Generate info for a dummy processing step
    dummy_sec_per_frame = 0.001
    dummy = {
        "slurm_params": {
            "mem": "4GB",
            "gpu": False,
            "sec_per_frame": dummy_sec_per_frame,
            "ncpus": 1,
            "jobs_in_progress": [],
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "dummy.py"),
            "conda_env": "dataPy_NWB2",
        },
        "func_args": {
            "video_path": "{video_path}",
        },
        "output_info": {
            "output_name": "dummy.npy",
        },
    }

    # Generate info for the calibration processing
    calib_sec_per_frame = (0.15 * 6) + 0.2  # 0.13 s/f is a conservative estimate for detection for 6 workers for one vid, times 6 vids per calibration, plus extra time for the extra steps of calibration
    calib = {
        "slurm_params": {
            "mem": "24GB",
            "gpu": False,
            "sec_per_frame": calib_sec_per_frame,
            "ncpus": 6,
            "jobs_in_progress": [],
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "calibration.py"),
            "conda_env": "dataPy_NWB2",  # TODO: make this dynamic
        },
        "func_args": {
            "video_dir": "{video_dir}",  # TODO: get these func args from a more reasonable location
        },
        "output_info": {
            "output_name": "camera_params.h5",  # saves an h5 file
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
            "jobs_in_progress": [],
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "segmentation.py"),
            "conda_env": "dataPy_torch2",  # TODO: make this dynamic?
        },
        "func_args": {  # NB: these must be in the right order here.
            "video_path": "{video_path}",
            "weights_path": find_files_from_pattern(
                weights_path, "centernet.pth"
            ),  # TODO: get these func args from a more reasonable location, ie the function should specify what its args are
        },
        "output_info": {
            "output_name": "centroid.npy",
        },
    }

    # hrnet params
    hrnet_sec_per_frame = 0.087
    hrnet = {
        "slurm_params": {
            "mem": "4GB",
            "gpu": True,
            "sec_per_frame": hrnet_sec_per_frame,
            "ncpus": 2,
            "jobs_in_progress": [],
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "detection.py"),
            "conda_env": "dataPy_torch2",  # TODO: make this dynamic?
        },
        "func_args": {  # NB: these must be in the right order here.
            "video_path": "{video_path}",
            "weights_dir": weights_path,  # TODO: get these func args from a more reasonable location, ie the function should specify what its args are
        },
        "output_info": {
            "output_name": "keypoints.h5",
        },
    }

    # triangulation params
    triangulation_sec_per_frame = 0.02
    triangulation = {
        "slurm_params": {
            "mem": "6GB",
            "gpu": False,
            "sec_per_frame": triangulation_sec_per_frame,
            "ncpus": 1,
            "jobs_in_progress": [],
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "triangulation.py"),
            "conda_env": "dataPy_NWB2",  # TODO: make this dynamic
        },
        "func_args": {
            "video_dir": "{video_dir}",
            "calib_file": "{calib_file}",
        },
        "output_info": {
            "output_name": "robust_triangulation.npy",
        },
    }

    # gimbal params
    gimbal_sec_per_frame = 0.09
    gimbal = {
        "slurm_params": {
            "mem": "8GB",
            "gpu": True,
            "sec_per_frame": gimbal_sec_per_frame,
            "ncpus": 1,
            "jobs_in_progress": [],
        },  
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "gimbal_smoothing.py"),
            "conda_env": "dataPy_KPMS_GIMBAL",  # TODO: make this dynamic
            "modules": ['gcc/9.2.0', 'ffmpeg', 'cuda/11.7'],
        },
        "func_args": {
            "vid_dir": "{video_dir}",
            "calib_file": "{calib_file}",
            "gimbal_params_file": gimbal_params_path,
        },
        "output_info": {
            "output_name": "gimbal.npy",
        }
    }

    other = {
        "video_dir": project_dir,  # assume config is in the dir with videos by default
        "verbose": False,
        "keypoint_colormap": "autumn",
    }

    sections = [
        ("VID_INFO", all_vid_info),
        ("SESSION_INFO", all_session_info),
        ("CALIBRATION", calib),
        ("CENTERNET", centernet),
        ("HRNET", hrnet),
        ("TRIANGULATION", triangulation),
        ("GIMBAL", gimbal),
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
    """Check that the config file is valid. This includes checking that
    paths exist and that conda environments exist.

    Parameters
    ----------
    config: str or dict
        The config file or a dictionary of the config file.

    Returns
    -------
    bool
    """

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
                    f"CONFIG ERROR: Path to weights does not exist: {check}"
                )

        check = config[top_level]["wrap_params"]["func_path"]
        if not os.path.exists(check):
            error_messages.append(
                f"CONFIG ERROR: Path to weights does not exist: {check}"
            )

        # Check that conda envs exist
        conda_env_folder = os.path.abspath(join(os.environ["CONDA_EXE"], "../../envs"))
        if not exists(
            join(conda_env_folder, config[top_level]["wrap_params"]["conda_env"])
        ):
            error_messages.append(
                f"CONFIG ERROR: Conda env does not exist: {config[top_level]['wrap_params']['conda_env']}"
            )

    # Check that all videos have same length
    # nframes = {k: v["nframes"] for k,v in config["VID_INFO"].items()}
    # if len(set([nf for nf in nframes.values()])) > 1:
    #     error_messages.append(
    #             f"CONFIG ERROR: Videos have different lengths: {nframes}"
    #         )

    if len(error_messages) == 0:
        return True
    for msg in error_messages:
        print(fill(msg, width=70, subsequent_indent="  "), end="\n\n")

    return False


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
    config_path = os.path.join(project_dir, config_name)

    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    if check_if_valid:
        check_config_validity(config)

    return config


def update_config(project_dir, new_config, verbose=True):
    """Update the config file.

    Parameters
    ----------
    project_dir: str
        Directory containing the config file

    new_config: dict
        Dictionary of new config values

    verbose: bool, default=True
        Print out the updated values

    Returns
    -------
    None
    """
    update_msgs = []
    import pdb

    def recursive_update(config, updates):
        for key, value in updates.items():
            # if key == "func_args" and "calib_file" in config[key]:
                # pdb.set_trace()
            if key in config and isinstance(config[key], dict):
                # If the key is a dictionary, recurse
                recursive_update(config[key], value)
            else:
                # Otherwise, update the value directly
                if verbose:
                    update_msgs.append(f"Updated {key} to {value}")
                config[key] = value
        return config

    config = load_config(project_dir, check_if_valid=False)
    config = recursive_update(config, new_config)
    if not check_config_validity(config):
        raise ValueError("Config is not valid after update!")
    else:
        if verbose:
            for msg in update_msgs:
                print(fill(msg, width=70, subsequent_indent="  "), end="\n\n")

    save_config(project_dir, config)

    return


def save_config(project_dir, config):
    """Save a dictionary as a yaml file.

    Parameters
    ----------
    project_dir: str
        Directory containing the config file

    config: dict
        Dictionary to save

    Returns
    -------
    None
    """
    sections = [(k, v) for k, v in config.items()]
    with open(os.path.join(project_dir, config_name), "w") as f:
        f.write(_build_yaml(sections, config_comments))
    # time.sleep(0.1)  # wait for file to be written
    return