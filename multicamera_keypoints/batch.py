import os
from os.path import join
import datetime
import time
import warnings

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


def prepare_calibration_batch(project_dir, overwrite=False):
    """Prepare a batch script for each step of the calibration process.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    overwrite : bool
        If True, overwrite existing data.

    Returns
    -------
    None
    """
    # Load the project config
    project_dir = os.path.abspath(project_dir)
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
                "proc_time": format_time_from_sec(time_sec),
            }}},
            verbose=False,
        )

    # Make a directory to hold files relevant to each proc step
    step_dir = join(project_dir, "keypoint_batch", "CALIBRATION")
    slurm_out_dir = join(step_dir, "slurm_outs")
    os.makedirs(step_dir, exist_ok=True)
    os.makedirs(slurm_out_dir, exist_ok=True)
    update_config(
        project_dir, 
        {"CALIBRATION": {"slurm_params": {
            "slurm_out_dir": slurm_out_dir,
            "step_dir": step_dir,
        }}},
        verbose=False,
    )

    # Generate the slurm scripts
    time.sleep(0.5)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config = load_config(project_dir)
    n_skipped_because_done = 0
    n_cmds = 0
    for vid_name, vid_info in config["CALIBRATION_VIDEOS"].items():

        # Check if this step is done on this vid, if so, skip
        if vid_info["CALIBRATION_done"] and not overwrite:
            n_skipped_because_done += 1
            continue

        slurm_cmd = _make_slurm_cmd(**config["CALIBRATION"]['slurm_params'], time=vid_info["proc_time"], slurm_out_prefix=f"calibration_{now}", job_name=f"{vid_name}_CALIBRATION")
        wrap_cmd = _make_wrap_cmd(**config["CALIBRATION"]['wrap_params'])
        cmd = slurm_cmd + wrap_cmd
        out_file = join(config["CALIBRATION"]["slurm_params"]["step_dir"], f"CALIBRATION_batch_{now}.sh")
        n_cmds += 1
        with open(out_file, "a") as f:
            args_str = " ".join([arg for arg in config["CALIBRATION"]["func_args"].values()]).format(video_dir=vid_info["video_dir"])
            full_cmd = cmd.format(args=args_str)
            f.write(full_cmd)
            f.write("\n\n")

    if n_cmds > 0:
        print(f"Batch script for calibration ready at {out_file}, containing {n_cmds} jobs.")
    if n_skipped_because_done:
        print(f"Skipped {n_skipped_because_done} jobs because they were already done.")

    return


def prepare_batch(project_dir, processing_steps=None, increment_time_fraction=1.0, overwrite=False):
    """Prepare a batch script for each processing step.

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

    processing_steps : list of str
        List of processing steps to run. 

    increment_time_fraction : float, optional
        If given, the time requested for each job is multiplied by this factor.
        Useful for when a few jobs fail due to timeout and you need to re-run.

    overwrite : bool
        If True, recalculate the number of frames in each video, re-do processing steps, etc.

    Returns
    -------
    None
    """
    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)

    if processing_steps is None:
        raise ValueError("Please specify at least one processing step to run")

    # Prep processing for each video
    for vid_name, vid_info in config["VID_INFO"].items():

        # If not already done, calculate the number of frames in each video
        if ("nframes" not in vid_info) or overwrite:
            nframes = vid_info["nframes"] = count_frames_cached(vid_info["video_path"])
        else:
            nframes = vid_info["nframes"]

        # Calculate how much time needed for each step of processing
        for step in processing_steps:
            time_sec = int(increment_time_fraction * max(5*60, nframes * config[step]["slurm_params"]["sec_per_frame"]))  # min 5 minutes
            vid_info[f"{step}_time"] = format_time_from_sec(time_sec)

    # Update the config
    update_config(
        project_dir,
        {"VID_INFO": {vid_name: vid_info for vid_name, vid_info in config["VID_INFO"].items()}},
        verbose=False,
    )

    # Make a directory to hold files relevant to each proc step
    for step in processing_steps:
        step_dir = join(project_dir, "keypoint_batch", step)
        slurm_out_dir = join(step_dir, "slurm_outs")
        os.makedirs(step_dir, exist_ok=True)
        os.makedirs(slurm_out_dir, exist_ok=True)
        update_config(
            project_dir, 
            {step: {"slurm_params": {
                "slurm_out_dir": slurm_out_dir,
                "step_dir": step_dir,
            }}},
            verbose=False,
        )

    # Generate the commands required for each step / video
    time.sleep(0.5)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = load_config(project_dir)
    n_skipped_because_not_ready = 0
    n_skipped_because_done = 0
    n_skipped_because_running = 0
    for step in processing_steps:
        n_cmds = 0
        for vid_name, vid_info in config["VID_INFO"].items():

            # Check all previous steps are done on this video
            previous_steps = PROCESSING_STEPS[:PROCESSING_STEPS.index(step)]
            if not all([f"{prev}_done" in vid_info and vid_info[f"{prev}_done"] for prev in previous_steps]):
                n_skipped_because_not_ready += 1
                continue

            # Check if this session is currently being run
            jobs_in_prog_names = [d["NAME"] for d in config[step]["slurm_params"]["jobs_in_progress"].values()]
            if any([vid_name in job_name for job_name in jobs_in_prog_names]):
                n_skipped_because_running += 1
                continue

            # Check if this step is done on this vid, if so, skip
            if f"{step}_done" in vid_info and vid_info[f"{step}_done"] and not overwrite:
                n_skipped_because_done += 1
                continue

            slurm_cmd = _make_slurm_cmd(
                time=vid_info[f"{step}_time"],
                **config[step]['slurm_params'],
                slurm_out_prefix=f"{step}_{vid_name}_{now}",
                job_name=f"{vid_name}_{step}",
            )
            wrap_cmd = _make_wrap_cmd(**config[step]['wrap_params'])
            cmd = slurm_cmd + wrap_cmd
            out_file = join(config[step]["slurm_params"]["step_dir"], f"{step}_batch_{now}.sh")
            with open(out_file, "a") as f:
                args_str = " ".join([arg for arg in config[step]["func_args"].values()]).format(**vid_info)
                full_cmd = cmd.format(args=args_str)
                f.write(full_cmd)
                f.write("\n\n")
            n_cmds += 1

        if n_cmds > 0:
            print(f"Batch script for step {step} ready, containing {n_cmds} jobs.")
            print("\t script: ", out_file)
        if n_skipped_because_not_ready:
            print(f"Skipped {n_skipped_because_not_ready} jobs because previous steps were not done.")
        if n_skipped_because_running:
            print(f"Skipped {n_skipped_because_running} jobs because they are already being processed.")
        if n_skipped_because_done:
            print(f"Skipped {n_skipped_because_done} jobs because they were already done")
        
    return


def prepare_session_batch(project_dir, processing_steps=None, increment_time_fraction=1.0, overwrite=False):
    """ Prepare a batch script for session-level processing steps, ie TRIANGULATION and GIMBAL.
    """
    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)

    if processing_steps is None:
        raise ValueError("Please specify at least one processing step to run")
    elif not all([step in ["TRIANGULATION", "GIMBAL"] for step in processing_steps]):
        raise ValueError("Only TRIANGULATION and GIMBAL are supported as session-level processing steps")

    # Prep processing for each video
    for session_name, session_info in config["SESSION_INFO"].items():

        # Confirm that the session is ready for processing
        # (for each video in the session, check in the video info that centernet and hrnet are done)
        if not all([all([config["VID_INFO"][video][f"{step}_done"] for step in ["CENTERNET", "HRNET"]]) for video in session_info["videos"]]):
            config["SESSION_INFO"][session_name]["ready_for_processing"] = False
        else:
            config["SESSION_INFO"][session_name]["ready_for_processing"] = True
        
        # Find nframes for each video in the session
        nframes = config["VID_INFO"][session_info["videos"][0]]["nframes"]

        # Calculate how much time needed for each step of processing
        for step in processing_steps:
            time_sec = int(increment_time_fraction * max(5*60, nframes * config[step]["slurm_params"]["sec_per_frame"]))  # min 5 minutes
            session_info[f"{step}_time"] = format_time_from_sec(time_sec)

    # Update the config
    update_config(
        project_dir,
        {"SESSION_INFO": config["SESSION_INFO"]},
        verbose=False,
    )

    # Make a directory to hold files relevant to each proc step
    for step in processing_steps:
        step_dir = join(project_dir, "keypoint_batch", step)
        slurm_out_dir = join(step_dir, "slurm_outs")
        os.makedirs(step_dir, exist_ok=True)
        os.makedirs(slurm_out_dir, exist_ok=True)
        update_config(
            project_dir, 
            {step: {"slurm_params": {
                "slurm_out_dir": slurm_out_dir,
                "step_dir": step_dir,
            }}},
            verbose=False,
        )

    # Generate the commands required for each step / video
    time.sleep(0.5)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = load_config(project_dir)
    n_skipped_because_not_ready = 0
    n_skipped_because_running = 0
    n_skipped_because_done = 0
    for step in processing_steps:
        n_cmds = 0
        for session_name, session_info in config["SESSION_INFO"].items():


            # Check triang is done before gimbal
            if step == "GIMBAL" and not session_info["TRIANGULATION_done"]:
                n_skipped_because_not_ready += 1
                continue
            
            if not session_info["ready_for_processing"]:
                n_skipped_because_not_ready += 1
                continue

            # Check if this step is done on this session, if so, skip
            if f"{step}_done" in session_info and session_info[f"{step}_done"] and not overwrite:
                n_skipped_because_done += 1
                continue

            # Check if this session is currently being run
            jobs_in_prog_names = [d["NAME"] for d in config[step]["slurm_params"]["jobs_in_progress"].values()]
            if any([session_name in job_name for job_name in jobs_in_prog_names]):
                n_skipped_because_running += 1
                continue

            slurm_cmd = _make_slurm_cmd(
                time=session_info[f"{step}_time"],
                **config[step]['slurm_params'],
                slurm_out_prefix=f"{step}_{session_name}_{now}",
                job_name=f"{session_name}_{step}",
            )
            wrap_cmd = _make_wrap_cmd(**config[step]['wrap_params'])
            cmd = slurm_cmd + wrap_cmd
            out_file = join(config[step]["slurm_params"]["step_dir"], f"{step}_batch_{now}.sh")
            with open(out_file, "a") as f:
                try:
                    args_str = " ".join([arg for arg in config[step]["func_args"].values()]).format(**session_info)
                except KeyError:
                    warnings.warn(f"Could not find all required arguments for step {step} in session {session_name}. Skipping.")
                    continue
                full_cmd = cmd.format(args=args_str)
                f.write(full_cmd)
                f.write("\n\n")
            n_cmds += 1

        if n_cmds > 0:
            print(f"Batch script for step {step} ready, containing {n_cmds} jobs.")
            print("\t script: ", out_file)
        else:
            print(f"No jobs ready for step {step}")
        if n_skipped_because_not_ready:
            print(f"Skipped {n_skipped_because_not_ready} sessions because previous steps were not done.")
        if n_skipped_because_running:
            print(f"Skipped {n_skipped_because_running} sessions because they are already being processed.")
        if n_skipped_because_done:
            print(f"Skipped {n_skipped_because_done} sessions because they were already done")
        

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
    """Run the most recent batch script for a processing step, or a user-specified script.

    Notes:
        -- This function assesses recency by file name, not by file modification time.
        -- This function only allows one set of jobs to be running at a time (could modify this in the future).

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
        top_level = processing_step
    else:
        shell_script_to_run = shell_script
        top_level = shell_script

    # # Make sure jobs aren't already running
    # if top_level in config and "jobs_in_progress" in config[top_level]["slurm_params"]:
    #     jip = config[top_level]["slurm_params"]["jobs_in_progress"]
    #     if jip:
    #         print(f"Jobs already running for {top_level}: {jip}")
    #         return

    # Run the shell script
    print(f"Running script {os.path.basename(shell_script_to_run)}")
    os.system(f'chmod +x {shell_script_to_run}')
    out = os.popen(f'{shell_script_to_run}').read()
    print(out)

    # Update the config with the list of running jobs
    # submitted_jobs = re.findall(r"Submitted batch job (\d+)", out)
    # update_config(project_dir, {top_level: {"slurm_params": {"jobs_in_progress": submitted_jobs}}})
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
        running_jobs = {k: v for k, v in running_jobs.items() if (processing_step in v['NAME'] and any([vid_name in v['NAME'] for vid_name in all_video_names]))}
    elif processing_step in ["TRIANGULATION", "GIMBAL"]:
        all_session_names = list(config["SESSION_INFO"].keys())
        running_jobs = {k: v for k, v in running_jobs.items() if (processing_step in v['NAME'] and any([session_name in v['NAME'] for session_name in all_session_names]))}

    # Update the config
    config[processing_step]["slurm_params"]["jobs_in_progress"] = running_jobs
    save_config(project_dir, config)

    return running_jobs


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

    import pdb

    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)
    output_name = config[processing_step]["output_info"]["output_name"]

    # Find the videos to be checked
    if processing_step in ["CENTERNET", "HRNET"]:
        section = "VID_INFO"
        def output_file_getter(vid_info):
            return vid_info["video_path"].replace("mp4", output_name)
    elif processing_step in ["TRIANGULATION", "GIMBAL"]:
        section = "SESSION_INFO"
        def output_file_getter(vid_info):
            return join(vid_info["video_dir"], os.path.basename(vid_info["video_dir"]) + "." + output_name)
    else:
        section = "CALIBRATION_VIDEOS"
        def output_file_getter(vid_info):
            return join(vid_info["video_dir"], os.path.basename(vid_info["video_dir"]) + "." + output_name)

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

    """Summarize the progress of each processing step.

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

    # Print the progress
    for step in ["CENTERNET", "HRNET"]:
        if step in config:
            n_done = sum([vid_info[f"{step}_done"] for vid_info in config["VID_INFO"].values()])
            n_total = len(config["VID_INFO"])
            print(f"{step}: {n_done}/{n_total} videos done")

    for step in ["TRIANGULATION", "GIMBAL"]:
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
        elif status == "TIMEOUT":
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
