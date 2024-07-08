import os
import time
from os.path import exists, join
from pathlib import Path
from textwrap import fill

import yaml
from o2_utils.selectors import find_files_from_pattern

import multicamera_keypoints.v0 as mkv0

# from multicamera_keypoints.file_utils import find_files_from_pattern
from multicamera_keypoints.vid_utils import count_frames_cached

PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
# PROCESSING_STEPS = ["CENTERNET", "HRNET", "TRIANGULATION", "GIMBAL"]  # not including calibration and whatnot

config_comments = {
    "slurm_params": "parameters for submitting jobs to slurm",
    "weights_path": "path to pretrained weights",
    "output_name": "suffix for the outputted npy file, ie video_base_name.output_name.npy",
    "conda_env": "name of the conda environment to use for running the code",
    "verbose": "whether to print progress messages during fitting",
    "keypoint_colormap": "colormap used for visualization; see `matplotlib.cm.get_cmap` for options",
}


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
            text = yaml.dump(
                {key: value}, default_flow_style=False, sort_keys=False
            ).strip()
            # Add comments if they exist for the key
            if key in comments:
                text = f"\n{'#'} {comments[key]}\n{text}"
            text_blocks.append(text)
    return "\n".join(text_blocks)


def add_videos_to_config(project_dir, video_paths, overwrite=False):
    # Find the existing videos
    config = load_config(project_dir)
    existing_vid_info = config["VID_INFO"]
    existing_session_info = config["SESSION_INFO"]

    # Get the processing steps in this config
    VID_PROCESSING_STEPS = [
        k
        for k in config
        if ("pipeline_info" in k and k["pipeline_info"]["processing_level"] == "video")
    ]
    SESSION_PROCESSING_STEPS = [
        k
        for k in config
        if (
            "pipeline_info" in k
            and (
                k["pipeline_info"]["processing_level"] == "session"
                or k["pipeline_info"]["processing_level"] == "calibration"
            )
        )
    ]

    # Remove videos that are already in the config from list of vids to add, if not overwriting
    if not overwrite:
        video_paths = {
            v
            for v in video_paths
            if os.path.basename(v.split(".mp4")[0]) not in existing_vid_info
        }

    if len(video_paths) == 0:
        print("No new videos to add to config")
        return

    # Get info about the new videos
    new_vid_info, new_session_info = _prepare_video_config(
        video_paths, VID_PROCESSING_STEPS, SESSION_PROCESSING_STEPS
    )

    # Update the config
    existing_vid_info.update(new_vid_info)
    existing_session_info.update(new_session_info)
    update_config(
        project_dir,
        {
            "VID_INFO": existing_vid_info,
            "SESSION_INFO": existing_session_info,
        },
        verbose=False,
    )

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
        calibration_paths = {
            v
            for v in calibration_paths
            if os.path.basename(v.split(".mp4")[0]) not in existing_calib_info
        }

    # Get info about the new calibrations
    _vid_info = _get_video_info(
        [
            find_files_from_pattern(p, "*.top*.mp4", n_expected=1, error_behav="raise")
            for p in calibration_paths
        ]
    )
    calib_info = {}
    for (calib_name, vid_info), calib_path in zip(_vid_info.items(), calibration_paths):
        short_name = calib_name.split(".")[0]
        calib_info[short_name] = {}
        calib_info[short_name]["video_dir"] = calib_path
        calib_info[short_name]["nframes"] = vid_info["nframes"]

    # Add key/val pairs specific to the calibration
    for vid_info in calib_info.values():
        vid_info.update(
            dict(
                CALIBRATION_done=False,
                ok=False,
                board_size=board_size,
                square_size=square_size,
            )
        )

    existing_calib_info.update(calib_info)
    update_config(
        project_dir, {"CALIBRATION_VIDEOS": existing_calib_info}, verbose=False
    )
    print(f"Added {len(calib_info)} calibration(s) to config")
    return


def _get_video_info(video_paths):
    """Parse a list of video paths into a dictionary with required info about each video.

    Uses a cached version of count_frames to speed up the process in case of errors.

    Parameters
    ----------
    video_paths: list of str
        A list of the full paths to video files.

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
            file_size = os.path.getsize(video)
            kb_per_fr = file_size / nframes / 1e3
        except Exception as e:
            print(f"Error counting frames in {video}, skipping...")
            print(f"Error: {e}")
            continue
        video_name, _ = os.path.splitext(os.path.basename(video))
        session_name = video_name.split(".")[0]
        vid_info[video_name] = {
            "video_path": os.path.abspath(video),
            "nframes": nframes,
            "session_name": session_name,
            "kb_per_fr": kb_per_fr,
        }
    return vid_info


def _prepare_video_config(video_paths, vid_steps, session_steps):
    # Get info about the new videos (just frame num and path for now)
    new_vid_info = _get_video_info(video_paths)

    # Add key/val pairs specific to each processing step
    for vid_info in new_vid_info.values():
        vid_info.update({f"{step}_done": False for step in vid_steps})

    # Parse videos into sessions
    sessions = [v["session_name"] for v in new_vid_info.values()]
    session_dirs = [os.path.dirname(v["video_path"]) for v in new_vid_info.values()]
    uq_pairings = list(set(list(zip(sessions, session_dirs))))
    new_session_info = {}
    for session, _dir in uq_pairings:
        new_session_info[session] = {
            "videos": [],
            "session_name": session, # redundant but useful
            "video_dir": _dir,
            "ready_for_processing": False,  # whether all videos are done with centernet and hrnet
            **{f"{step}_done": False for step in session_steps},
        }

    # Add list of videos to each session
    for vid_name in new_vid_info:
        session = new_vid_info[vid_name]["session_name"]
        new_session_info[session]["videos"].append(vid_name)

    # Check whether all videos in a session have the same num frames
    for session in new_session_info:
        vids = new_session_info[session]["videos"]
        nframes = [new_vid_info[v]["nframes"] for v in vids]
        if len(set(nframes)) > 1:
            print(
                f"Warning: Videos in session {session} have different lengths; cross-video frame alignment will be required."
            )
            new_session_info[session]["alignment_required"] = True
        else:
            new_session_info[session]["alignment_required"] = False

    return new_vid_info, new_session_info


def generate_config(
    project_dir,
    video_paths,
    weights_path,
    gimbal_params_path,
    non_default_config=None,
    overwrite=False,
    config_name="keypoint_config.yml",
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

    config_name: str, optional
        Name of the config file to generate.

    Returns
    -------
    None
    """

    # Prevent overwriting existing config file
    config_path = join(project_dir, config_name)
    if exists(config_path) and not overwrite:
        raise ValueError(
            f"Config file already exists at {config_path}. Use `overwrite=True` to overwrite."
        )

    # Generate info for the calibration processing
    sections = []

    step_config, step_name = mkv0.calibration.make_config(PACKAGE_DIR)
    sections.append((step_name, step_config))

    # Generate info for the centernet
    step_config, step_name = mkv0.segmentation.make_config(PACKAGE_DIR, weights_path)
    sections.append((step_name, step_config))

    # hrnet params
    step_config, step_name = mkv0.detection.make_config(PACKAGE_DIR, weights_path)
    sections.append((step_name, step_config))

    # triangulation params
    step_config, step_name = mkv0.triangulation.make_config(PACKAGE_DIR)
    sections.append((step_name, step_config))

    # gimbal params
    step_config, step_name = mkv0.gimbal_smoothing.make_config(
        PACKAGE_DIR, gimbal_params_path
    )
    sections.append((step_name, step_config))

    # Get video info
    # all_vid_info = _get_video_info(video_paths)

    # # Add key/val pairs specific to each processing step
    # for vid_info in all_vid_info.values():
    #     vid_info.update({"CENTERNET_done": False, "HRNET_done": False})

    # # Parse videos into sessions
    # sessions = [v["session_name"] for v in all_vid_info.values()]
    # session_dirs = [os.path.dirname(v["video_path"]) for v in all_vid_info.values()]
    # uq_pairings = list(set(list(zip(sessions, session_dirs))))
    # all_session_info = {}
    # for session, _dir in uq_pairings:
    #     all_session_info[session] = {
    #         "videos": [],
    #         "CALIBRATION_done": False,
    #         "TRIANGULATION_done": False,
    #         "GIMBAL_done": False,
    #         "video_dir": _dir,
    #         "ready_for_processing": False,  # whether all videos are done with centernet and hrnet
    #     }

    # # Add list of videos to each session
    # for vid_name in all_vid_info:
    #     s = all_vid_info[vid_name]["session_name"]
    #     all_session_info[s]["videos"].append(vid_name)

    # Misc other required info
    other = {
        "video_dir": project_dir,  # assume config is in the dir with videos by default
        "verbose": False,
        "keypoint_colormap": "autumn",
    }
    sections.append(("OTHER", other))

    # Write the config file
    with open(config_path, "w") as f:
        f.write(_build_yaml(sections, config_comments))

    # Update the config with any non-default settings
    config = load_config(project_dir)
    if non_default_config is not None:
        update_config(config, non_default_config)

    # Add videos to the config!
    time.sleep(1)  # wait for file to be written
    config = load_config(project_dir)
    VID_PROCESSING_STEPS = [
        k
        for k in config
        if (
            "pipeline_info" in config[k]
            and config[k]["pipeline_info"]["processing_level"] == "video"
        )
    ]
    SESSION_PROCESSING_STEPS = [
        k
        for k in config
        if (
            "pipeline_info" in config[k]
            and (
                config[k]["pipeline_info"]["processing_level"] == "session"
                or config[k]["pipeline_info"]["processing_level"] == "calibration"
            )
        )
    ]
    all_vid_info, all_session_info = _prepare_video_config(
        video_paths, VID_PROCESSING_STEPS, SESSION_PROCESSING_STEPS
    )
    update_config(
        project_dir,
        {"VID_INFO": all_vid_info, "SESSION_INFO": all_session_info},
        verbose=False,
    )

    return


def add_section_to_config(
    project_dir,
    section_name,
    section,
    processing_level="videos",
    overwrite=False,
    config_name="keypoint_config.yml",
):
    """Manually add a new section to the config file.
    Useful for adding new processing steps to the config file,
    i.e. if you finetune a network and want to re-run from where you left off.

    Parameters
    ----------
    project_dir: str
        The directory containing the config file.

    section_name: str
        The name of the section to add.

    section: dict
        The section to add.

    processing_level: str, default="videos"
        Whether this processing step is done per video ("videos"), or per session ("sessions")


    overwrite: bool, default=False
        Whether to overwrite an existing section with the same name.
    Returns
    -------
    None
    """
    config = load_config(project_dir)
    if section_name in config and not overwrite:
        raise ValueError(f"Section {section_name} already exists in the config file.")

    config[section_name] = section

    if processing_level == "video":
        for vid_info in config["VID_INFO"].values():
            vid_info.update({f"{section_name}_done": False})
    elif processing_level == "session":
        for session_info in config["SESSION_INFO"].values():
            session_info.update({f"{section_name}_done": False})
    else:
        raise ValueError(f"processing_level must be 'video' or 'session' but got {processing_level}")

    save_config(project_dir, config)
    config_path = os.path.join(project_dir, config_name)
    check_config_validity(config_path)

    return


def check_config_validity(config=None, project_dir=None):
    """Check that the config file is valid. This includes checking that
    paths exist and that conda environments exist.

    Parameters
    ----------
    config: str or dict
        The config file or a dictionary of the config file.

    project_dir: str
        The directory containing the config file.

    Returns
    -------
    bool
    """

    if project_dir is not None and config is not None:
        raise ValueError("Only one of `config` and `project_dir` can be specified.")

    if project_dir is not None:
        config = load_config(project_dir)
    elif isinstance(config, str):
        with open(config, "r") as stream:
            config = yaml.safe_load(stream)
    elif not isinstance(config, dict):
        raise ValueError(
            "config must be a dictionary or a path to a config file, or pass project_dir."
        )

    error_messages = []

    # Check that required paths exist
    top_level_base_names = ["CENTERNET", "HRNET", "TRIANGULATION", "GIMBAL"]
    top_levels = [
        section_name
        for section_name in config.keys()
        if any([base_name in section_name for base_name in top_level_base_names])
    ]
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


def load_config(project_dir, check_if_valid=True, config_name="keypoint_config.yml"):
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

    def recursive_update(config, updates):
        for key, value in updates.items():
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


def save_config(project_dir, config, config_name="keypoint_config.yml"):
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
