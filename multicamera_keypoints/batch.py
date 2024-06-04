import os
from os.path import join
import datetime
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from multicamera_keypoints.vid_utils import count_frames_cached
from multicamera_keypoints.io import load_config, update_config, save_config, PROCESSING_STEPS
from multicamera_keypoints.file_utils import is_file_openable_and_contains_data
# from multicamera_keypoints.file_utils import find_files_from_pattern
from o2_utils.selectors import find_files_from_pattern
from o2_utils.slurm import get_job_info


def format_time_from_sec(seconds):
    """Convert seconds to HH:MM:SS format

    Parameters
    ----------
    seconds : int
        Duration in seconds

    Returns
    -------
    str
        The duration represented in HH:MM:SS format
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def calculate_seconds_from_time_str(time_str):
    """Convert HH:MM:SS format to seconds

    Parameters
    ----------
    time_str : str
        Duration in HH:MM:SS format

    Returns
    -------
    int
        The duration in seconds
    """
    hours, minutes, seconds = map(int, time_str.split(":"))
    return hours*3600 + minutes*60 + seconds


def _prepare_calibrations_for_batch(project_dir, overwrite=False):
    config = load_config(project_dir)

    # Prep processing for each video
    for vid_name, vid_info in config["CALIBRATION_VIDEOS"].items():

        # If not already done, calculate the number of frames in each video
        if ("nframes" not in vid_info) or overwrite:
            nframes = vid_info["nframes"] = count_frames_cached(find_files_from_pattern(vid_info["video_dir"], "*.top*.mp4", ["azure"]))
        else:
            nframes = vid_info["nframes"]

        # Calculate how much time needed for each step of processing
        time_sec = max(5*60, int(nframes * config["CALIBRATION"]["slurm_params"]["sec_per_frame"]))  # min 5 minutes
        update_config(
            project_dir, 
            {"CALIBRATION_VIDEOS": {vid_name: {
                "nframes": nframes,
                "CALIBRATION_time": format_time_from_sec(time_sec),
            }}},
            verbose=False,
        )


def _prepare_videos_for_batch(project_dir, processing_step, increment_time_fraction=1.0, recalculate=False):
    """Calculate the number of frames in each video and the time needed for each step of processing.

    Parameters
    ----------
    project_dir : str   
        Path to the project directory.

    processing_step: str
        The processing step to prepare for.

    recalculate : bool
        If True, recalculate the number of frames in each video.

    Returns
    -------
    None
    """

    config = load_config(project_dir)

    # Prep processing for each video
    for vid_name, vid_info in config["VID_INFO"].items():

        # If not already done, calculate the number of frames in each video
        if ("nframes" not in vid_info) or recalculate:
            nframes = vid_info["nframes"] = count_frames_cached(vid_info["video_path"])
        else:
            nframes = vid_info["nframes"]

        # Do a spot check for video issues by ensuring nframes is reasonable
        if nframes == 0:
            raise ValueError(f"Unable to count frames for video {vid_name}. The file may be corrupted — please check it manually.")
        elif nframes < 120:
            warnings.warn(f"Video {vid_name} has fewer than 120 frames. This may be an issue.")

        # Calculate how much time needed for each step of processing
        time_sec = int(increment_time_fraction * max(5*60, nframes * config[processing_step]["slurm_params"]["sec_per_frame"]))  # min 5 minutes
        vid_info[f"{processing_step}_time"] = format_time_from_sec(time_sec)

    # Update the config
    update_config(
        project_dir,
        {"VID_INFO": {vid_name: vid_info for vid_name, vid_info in config["VID_INFO"].items()}},
        verbose=False,
    )

    return


def _prepare_sessions_for_batch(project_dir, processing_step, increment_time_fraction=1.0, recalculate=False):
    
    config = load_config(project_dir)

    # Prep processing for each video
    for session_name, session_info in config["SESSION_INFO"].items():

        ### Confirm that the session is ready for processing  ###
        # for each video in the session, check in the video info that centernet and hrnet are done
        ready_for_processing = True
        for video in session_info["videos"]:
            for step in ["CENTERNET", "HRNET"]:
                if not all([config["VID_INFO"][video][f"{step}_done"]]):
                    ready_for_processing = False
                    
        # Check that the calibration file 1) exists 2) has been processed
        if session_info["calibration"] is None:
            ready_for_processing = False  # no calibration file has been matched to this session yet
        elif session_info["calibration"] is not None:
            calib_info = config["CALIBRATION_VIDEOS"][session_info["calibration"]]
            session_info["CALIBRATION_done"] = calib_info["CALIBRATION_done"]
            if not calib_info["CALIBRATION_done"]:
                ready_for_processing = False

        # Set the ready_for_processing flag
        config["SESSION_INFO"][session_name]["ready_for_processing"] = ready_for_processing
        
        # Find nframes for each video in the session
        nframes = config["VID_INFO"][session_info["videos"][0]]["nframes"]

        # Calculate how much time needed for each step of processing
        time_sec = int(increment_time_fraction * max(5*60, nframes * config[processing_step]["slurm_params"]["sec_per_frame"]))  # min 5 minutes
        session_info[f"{processing_step}_time"] = format_time_from_sec(time_sec)

    # Update the config
    update_config(
        project_dir,
        {"SESSION_INFO": config["SESSION_INFO"]},
        verbose=False,
    )


def prepare_batch(project_dir, processing_step, increment_time_fraction=1.0, overwrite=False):
    """Prepare a batch script for a processing step.

    This function is fairly agnostic to the specific processing step going on.
    It takes care of:
        -- figuring out how long to run the job for
        -- making sure your directories are set up correctly
        -- generating an sbatch command for each video.

    The step-specific code is currently in io.generate_config(),
    where, for each step, there is a hard-coded python script
    and most of their arguments.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    processing_step : str
        Single processing step to run. 

    increment_time_fraction : float, optional
        If given, the time requested for each job is multiplied by this factor.
        Useful for when a few jobs fail due to timeout and you need to re-run.

    overwrite : bool
        If True, pass " --overwrite" to the processing script.

    Returns
    -------
    None
    """
    
    config = load_config(project_dir)
    proc_level = config[processing_step]["pipeline_info"]["processing_level"]

    if proc_level == "calibration":
        _prepare_calibrations_for_batch(project_dir, overwrite=overwrite)
        config_info_key = "CALIBRATION_VIDEOS"

    elif proc_level == "video":
        _prepare_videos_for_batch(project_dir, processing_step=processing_step, increment_time_fraction=increment_time_fraction, recalculate=False)
        config_info_key = "VID_INFO"

    elif proc_level == "session":
        _prepare_sessions_for_batch(project_dir, processing_step=processing_step, increment_time_fraction=increment_time_fraction, recalculate=False)
        config_info_key = "SESSION_INFO"


    # Update cuarrently running jobs, in case any have finished since the last time we checked
    update_running_jobs(project_dir, processing_step)
        
    # Make a directory to hold files relevant to each proc step
    step_dir = join(project_dir, "keypoint_batch", processing_step)
    slurm_out_dir = join(step_dir, "slurm_outs")
    os.makedirs(step_dir, exist_ok=True)
    os.makedirs(slurm_out_dir, exist_ok=True)
    update_config(
        project_dir, 
        {processing_step: {"slurm_params": {
            "slurm_out_dir": slurm_out_dir,
            "step_dir": step_dir,
        }}},
        verbose=False,
    )

    # Generate the commands required for each step / video
    time.sleep(0.5)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = load_config(project_dir)
    n_skipped_because_prev_steps_not_done = 0
    n_skipped_because_not_ready = 0
    n_skipped_because_done = 0
    n_skipped_because_running = 0
    
    n_cmds = 0

    # For each video / session
    for item_name, item_info in config[config_info_key].items():

        # Check all previous steps are done on this item
        previous_steps = config[processing_step]["step_dependencies"]
        if previous_steps is not None and not all([f"{prev}_done" in item_info and item_info[f"{prev}_done"] for prev in previous_steps]):
            n_skipped_because_prev_steps_not_done += 1
            continue

        if "ready_for_processing" in item_info and not item_info["ready_for_processing"]:
            n_skipped_because_not_ready += 1
            continue

        # Check if this item is currently being processed
        jobs_in_prog_names = [d["NAME"] for d in config[processing_step]["slurm_params"]["jobs_in_progress"].values()]
        if any([item_name in job_name for job_name in jobs_in_prog_names]):
            n_skipped_because_running += 1
            continue

        # Check if this step is done on this item, if so, skip
        if f"{processing_step}_done" in item_info and item_info[f"{processing_step}_done"] and not overwrite:
            n_skipped_because_done += 1
            continue

        # Make the command to get an sbatch job with the right time / partition / etc.
        slurm_cmd = _make_slurm_cmd(
            time=item_info[f"{processing_step}_time"],
            **config[processing_step]['slurm_params'],
            slurm_out_prefix=f"{processing_step}_{item_name}_{now}",
            job_name=f"{item_name}_{processing_step}",
        )

        # Format the python command with conda env / modules / python script to use.
        wrap_cmd = _make_wrap_cmd(**config[processing_step]['wrap_params'])

        # Creat the full command
        cmd = slurm_cmd + wrap_cmd

        # Prepare an output file
        out_file = join(config[processing_step]["slurm_params"]["step_dir"], f"{processing_step}_batch_{now}.sh")
        
        with open(out_file, "a") as f:
            args_str = " ".join([arg for arg in config[processing_step]["func_args"].values()]).format(**item_info)
            if "func_kwargs" in config[processing_step]:
                kwargs_str = " ".join([f"--{k} {v}" if not isinstance(v, bool) else f"--{k}" for k, v in config[processing_step]["func_kwargs"].items()])
            else:
                kwargs_str = ""
            if "output_name" in config[processing_step]["output_info"]:
                kwargs_str = kwargs_str + f" --output_name {config[processing_step]['output_info']['output_name']}"
            if overwrite:
                args_str = args_str + " --overwrite"
            full_cmd = cmd.format(args=args_str, kwargs=kwargs_str)

            # Finally, add item-specific arguments (ie path to a specific video)
            full_cmd = cmd.format(args=args_str)  # TODO: this could probably be part of _make_wrap_cmd

            # Write to file
            f.write(full_cmd)
            f.write("\n\n")
        n_cmds += 1

    # Report results
    if n_cmds > 0:
        print(f"Batch script for processing_step {processing_step} ready, containing {n_cmds} jobs.")
        print("\t script: ", out_file)

    if n_skipped_because_prev_steps_not_done:
        print(f"Skipped {n_skipped_because_prev_steps_not_done} jobs because previous steps were not done.")
    if n_skipped_because_not_ready:
        print(f"Skipped {n_skipped_because_not_ready} jobs because session was not ready.")
    if n_skipped_because_running:
        print(f"Skipped {n_skipped_because_running} jobs because they are already being processed.")
    if n_skipped_because_done:
        print(f"Skipped {n_skipped_because_done} jobs because they were already done")
        
    return


def _make_slurm_cmd(
    time,
    gpu,
    mem,
    ncpus,
    job_name,
    slurm_out_dir,
    slurm_out_prefix,
    **kwargs,
):
    """Generate an sbatch command for slurm.

    Parameters
    ----------
    time : str
        Time to request for the job, in HH:MM:SS format

    gpu : bool
        Whether to request a gpu

    mem : str
        Memory to request for the job, e.g. "8GB"

    ncpus : int
        Number of cpus to request for the job

    job_name : str
        Name of the job. For example, "20240101_J04301_CENTERNET".

    slurm_out_dir : str
        Path to the directory to save slurm output files

    slurm_out_prefix : str
        Prefix for the slurm output files

    Returns
    -------
    slurm_str : str
        The sbatch command
    """

    if gpu:
        partition = "gpu_quad"
    else:
        partition = "short"

    slurm_str = (
        'sbatch '
        f'-p {partition} '
        f'-t {time} '
        f'--mem {mem} '
        f'-c {ncpus} '
        f'-J {job_name} '
        f'-o {slurm_out_dir}/slurm_{slurm_out_prefix}_%j.out '  # %j here is jobid, not job name
    )
    
    if gpu:
        slurm_str = slurm_str + " --gres=gpu:1 "
    
    slurm_str = slurm_str + '--wrap '
    
    return slurm_str


def _make_wrap_cmd(
    func_path,
    conda_env,
    modules=['gcc/9.2.0', 'ffmpeg'],
):
    """Generate the wrap portion of an sbatch command.

    Parameters
    ----------
    func_path : str
        Path to the function to run

    conda_env : str
        Name of the conda environment to activate

    modules : list, optional
        List of modules to load, by default ['gcc/9.2.0', 'ffmpeg']

    Returns
    -------
    wrap_str : str
        The wrap portion of the sbatch command
    """

    module_load_str = " ".join([f"module load {module};"for module in modules])
    wrap_str = (
        '\'eval "$(conda shell.bash hook)"; '
        f'conda activate {conda_env}; '
        f'{module_load_str} '
        f'python {func_path} {{args}}\''
    )
    
    return wrap_str


def run_batch(project_dir, processing_step, shell_script=None):
    """Run the most recent batch script for a processing step, or a user-specified script. Does NOT check if the specific item / step is already running.

    Notes:
        -- Automatically runs update_running_jobs() at the end.
        -- This function assesses recency by the datestring in the file name, not by file modification time.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    processing_step : str
        Step of keypoint processing to run

    shell_script : str, optional
        If given, the script at this path is run directly, instead of looking for the most recent script (sorted alphabetically).

    Raises
    ------
    ValueError
        If the processing step is not recognized

    Returns
    -------
    None
    """
    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)

    if shell_script is None:
        # Find the latest slurm script in the directory
        shell_scripts = find_files_from_pattern(
            config[processing_step]["slurm_params"]["step_dir"],
            f"{processing_step}_batch_*.sh",
            error_behav="pass")
        if isinstance(shell_scripts, list):
            shell_script_to_run = sorted(shell_scripts)[-1]
        elif isinstance(shell_scripts, str):
            shell_script_to_run = shell_scripts
    else:
        shell_script_to_run = shell_script

    # Run the shell script
    print(f"Running script {os.path.basename(shell_script_to_run)}")
    os.system(f'chmod +x {shell_script_to_run}')
    out = os.popen(f'{shell_script_to_run}').read()
    print(out)

    # Update the config with the list of running jobs
    time.sleep(5)
    update_running_jobs(project_dir, processing_step)

    return


def cancel_batch(project_dir, processing_step, shell_script=None):
    """Cancel the most recent batch script for a processing step, or a user-specified script.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    processing_step : str
        Step of keypoint processing to cancel

    shell_script : str, optional
        If given, this script is cancelled directly, instead of looking for the most recent script.

    Raises
    ------
    ValueError
        If the processing step is not recognized

    Returns
    -------
    None
    """
    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)
    if shell_script is None:
        top_level = processing_step
    else:
        top_level = shell_script

    # Cancel the jobs
    for job in config[top_level]["slurm_params"]["jobs_in_progress"]:
        os.system(f'scancel {job}')
        print(f"Cancelled {job}")

    # Update the config
    time.sleep(5)
    update_running_jobs(project_dir, processing_step)

    return


def parse_squeue_output(output):
    """Parse the output of the squeue command into a dict with headers / values.

    """
    lines = output.strip().split('\n')
    headers = lines[0].split()
    job_data = {}
    
    for line in lines[1:]:
        fields = line.split(None, len(headers) - 1)
        job_info = dict(zip(headers, fields))
        job_id = job_info['JOBID']
        job_data[job_id] = job_info
    
    return job_data


def update_running_jobs(project_dir, processing_step):
    """Update the list of running jobs for a processing step.

    Note: this command calls squeue, so it can take a little while.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    processing_step : str
        Step of keypoint processing to update

    Returns
    -------
    None
    """

    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)

    # Get the list of running jobs
    running_jobs = parse_squeue_output(os.popen("squeue --me -o '%.18i %.9P %.50j %.8u %.2t %.10M %.9l %.6D %R'").read())
    
    if processing_step in ["CENTERNET", "HRNET"]:
        all_video_names = list(config["VID_INFO"].keys())    
        running_jobs = {k: v for k, v in running_jobs.items() if (processing_step in v['NAME'] and any([vid_name in v['NAME'] for vid_name in all_video_names]))}  # TODO: fix, rn this will detect HRNET.finetune as belonging to both "HRNET.finetune" and "HRNET" steps
    elif processing_step in ["TRIANGULATION", "GIMBAL"]:
        all_session_names = list(config["SESSION_INFO"].keys())
        running_jobs = {k: v for k, v in running_jobs.items() if (processing_step in v['NAME'] and any([session_name in v['NAME'] for session_name in all_session_names]))}

    # Update the config
    config[processing_step]["slurm_params"]["jobs_in_progress"] = running_jobs
    save_config(project_dir, config)

    return running_jobs


def verify_compression_outputs_by_size(project_dir, processing_step, overwrite=False):
    """Verify that compression worked by comparing size of video per frame.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    overwrite : bool
        If True, re-check videos even if they are marked as done in the config.

    Returns
    -------
    None
    """
    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)

    if not any(["COMPRESSION" in step for step in config.keys()]):
        raise ValueError("No compression step found in config.")
    elif processing_step not in config.keys():
        raise ValueError(f"Processing step {processing_step} not found in config.")
    elif "COMPRESSION" not in processing_step:
        raise ValueError(f"Processing step {processing_step} is not a compression step.")

    # The compression section of the config has an attr under output_info called "expected_post_comp_max_kb_per_frame"
    # This is the expected maximum size of the video after compression, in kb per frame.
    # Use this to verify that compression worked as expected -- if the size of the video per frame is less than this, it probably worked.
    expected_kb_per_fr = config[processing_step]["output_info"]["expected_post_comp_max_kb_per_frame"]

    n_compressed = 0
    n_not_compressed = 0
    comprn_key = f"{processing_step}_done"

    for vid_name, vid_info in config["VID_INFO"].items():
        if vid_info[comprn_key] and not overwrite:
            continue
    

        actual_kb_per_fr = os.path.getsize(vid_info["video_path"]) / vid_info["nframes"] / 1024

        if actual_kb_per_fr < expected_kb_per_fr:
            n_compressed += 1
            vid_info[comprn_key] = True
        else:
            n_not_compressed += 1
            vid_info[comprn_key] = False

    # Update the config
    update_config(project_dir, {"VID_INFO": config["VID_INFO"]}, verbose=False)

    print(f"{n_compressed} videos compressed, {n_not_compressed} videos not compressed")

    return


def verify_outputs(project_dir, processing_step, overwrite=False):
    """Verify that the outputs of a processing step are present and complete.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    processing_step : str
        Step of keypoint processing to verify

    overwrite : bool
        If True, re-check videos even if they are marked as done in the config.

    Returns
    -------
    None
    """

    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)
    output_name = config[processing_step]["output_info"]["output_name"]

    # Find the videos to be checked
    if any([base_step in processing_step  for base_step in ["CENTERNET", "HRNET"]]):
        section = "VID_INFO"
        def output_file_getter(vid_info):
            return vid_info["video_path"].replace("mp4", output_name)
    elif any([base_step in processing_step for base_step in ["TRIANGULATION", "GIMBAL"]]):
        section = "SESSION_INFO"
        def output_file_getter(vid_info):
            return join(vid_info["video_dir"], os.path.basename(vid_info["video_dir"]) + "." + output_name)
    elif processing_step == "CALIBRATION":
        section = "CALIBRATION_VIDEOS"
        def output_file_getter(vid_info):
            return join(vid_info["video_dir"], os.path.basename(vid_info["video_dir"]) + "." + output_name)
    else:
        raise ValueError("Processing step not recognized")

    good_files = []
    incomplete_files = []
    missing_files = []
    for _, vid_info in config[section].items():

        output_file = output_file_getter(vid_info)

        # Check if already known to be done
        if vid_info[f"{processing_step}_done"] and not overwrite:
            good_files.append(output_file)
            continue

        if not os.path.exists(output_file):
            missing_files.append(output_file)
            vid_info[f"{processing_step}_done"] = False
        elif not is_file_openable_and_contains_data(output_file):
            incomplete_files.append(output_file)
            vid_info[f"{processing_step}_done"] = False
        else:
            good_files.append(output_file)
            vid_info[f"{processing_step}_done"] = True

    # Print the results
    print(f"{len(good_files)} out of {len(config[section])} files look ok.")
    if missing_files:
        print(f"{len(missing_files)} files are not present")
    if incomplete_files:
        print(f"{len(incomplete_files)} files are incomplete")

    # Update the config
    update_config(project_dir, {section: config[section]}, verbose=False)

    return good_files, incomplete_files, missing_files


def summarize_progress(project_dir):
    """Summarize the progress of each processing step. (Only looks at config, does not actually check files.)

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    Returns
    -------
    None
    """
    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)

    all_steps = config.keys()

    # Print the progress
    vid_level_steps = [step for step in all_steps if any([base_step in step for base_step in ["CENTERNET", "HRNET"]])]
    for step in vid_level_steps:
        if step in config:
            n_done = sum([vid_info[f"{step}_done"] for vid_info in config["VID_INFO"].values()])
            n_total = len(config["VID_INFO"])
            print(f"{step}: {n_done}/{n_total} videos done")

    session_level_steps = [step for step in all_steps if any([base_step in step for base_step in ["TRIANGULATION", "GIMBAL"]])]
    for step in session_level_steps:
        if step in config:
            n_done = sum([vid_info[f"{step}_done"] for vid_info in config["SESSION_INFO"].values()])
            n_total = len(config["SESSION_INFO"])
            print(f"{step}: {n_done}/{n_total} sessions done")

    return


def evaluate_failed_jobs(jobids):
    """For a list of jobids, evaluate if they failed, and if so, try to determine why.

    Parameters
    ----------
    jobids : list of str
        List of jobids to evaluate

    Returns
    -------
    job_info : dict
        Dictionary containing information about the jobs
    
    failed_jobs : list of str
        List of jobids that failed

    timedout_jobs : list of str
        List of jobids that timed out
    """
    job_info = {}
    failed_jobs = []
    timedout_jobs = []
    for jobid in jobids:
        job_info[jobid] = {"sacct_info": get_job_info(jobid)}
        status = job_info[jobid]["sacct_info"]["Status"]
        if status == "FAILED":
            failed_jobs.append(jobid)
        elif status == "TIMEOUT" or status == "CANCELLED":
            timedout_jobs.append(jobid)

    print(f"{len(failed_jobs)} failed: {failed_jobs}")
    print(f"{len(timedout_jobs)} timed out: {timedout_jobs}")

    return job_info, failed_jobs, timedout_jobs


def evaluate_resource_usage(jobids, plot=True):
    """Plot the time and memory usage for a list of jobs. Only considers COMPLETED jobs.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    jobids : list of str
        List of jobids to evaluate

    plot : bool
        If True, plot the results
    
    """

    job_info = {}
    for jobid in jobids:
        job_info[jobid] = {"sacct_info": get_job_info(jobid)}

        if job_info[jobid]["sacct_info"]["Status"] != "COMPLETED":
            job_info.pop(jobid)
            continue

        # import pdb
        # pdb.set_trace()

        # Calculate fraction of time used
        elapsed_time = job_info[jobid]["sacct_info"]["ElapsedTime"]
        requested_time = job_info[jobid]["sacct_info"]["RequestedTime"]
        job_info[jobid]["time_fraction"] = calculate_seconds_from_time_str(elapsed_time) / calculate_seconds_from_time_str(requested_time)

        # Calculate fraction of memory used
        mem_req = job_info[jobid]["sacct_info"]["Mem_Req"]
        mem_used = job_info[jobid]["sacct_info"]["Mem_Used"]
        job_info[jobid]["mem_fraction"] = (
            float(mem_used[:-1]) / float(mem_req[:-1])
        )

    if plot:
        # Generate a cute little summary figure
        # Figure with four subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs = axs.flatten()

        # sort the jobids by time usage
        sorted_jobids = sorted(job_info, key=lambda x: job_info[x]["time_fraction"])
        time_usages = [job_info[jobid]["time_fraction"] for jobid in sorted_jobids]
        mem_usages_sorted_wrt_time = [job_info[jobid]["mem_fraction"] for jobid in sorted_jobids]
        axs[0].bar(sorted_jobids, time_usages)
        axs[0].set_title("Sorted fractional time usage")
        axs[1].bar(sorted_jobids, mem_usages_sorted_wrt_time)
        axs[1].set_title("[memory]")
        
        # sort the jobids by memory usage, and plot
        sorted_jobids = sorted(job_info, key=lambda x: job_info[x]["mem_fraction"])
        mem_usages = [job_info[jobid]["mem_fraction"] for jobid in sorted_jobids]
        time_usages_sorted_wrt_mem = [job_info[jobid]["time_fraction"] for jobid in sorted_jobids]
        axs[2].bar(sorted_jobids, mem_usages)
        axs[2].set_title("Sorted fractional memory usage")
        axs[3].bar(sorted_jobids, time_usages_sorted_wrt_mem)
        axs[3].set_title("[time]")

        for ax in axs:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel("JobID")
            ax.set_ylabel("Fraction used")
            ax.set_ylim([0, 1])

        fig.tight_layout()

    return job_info
