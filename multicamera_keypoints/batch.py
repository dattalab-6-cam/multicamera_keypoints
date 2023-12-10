import tqdm
import numpy as np
import h5py
import os
from os.path import join, exists
import sys
import cv2
from vidio.read import OpenCVReader
import matplotlib.pyplot as plt
import datetime
import re
import time

from multicamera_keypoints.vid_utils import count_frames
from multicamera_keypoints.io import load_config, update_config
from multicamera_keypoints.file_utils import find_files_from_pattern


def format_time_from_sec(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def prepare_batch(project_dir, processing_steps=None, recalculate=False):
    # Load the project config
    project_dir = os.path.abspath(project_dir)
    config = load_config(project_dir)

    if processing_steps is None:
        raise ValueError("Please specify at least one processing step to run")

    # Calculate how much time needed for each session
    if "nframes" not in config["VID_INFO"]["top"] or recalculate:
        nframes = count_frames(config["VID_INFO"]["top"]["video_path"])
    else:
        nframes = config["VID_INFO"]["top"]["nframes"]


    # processing_steps = ["CENTERNET"]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate how much time needed for each step of processing
    for step in processing_steps:
        time_sec = max(5*60, int(nframes * config[step]["slurm_params"]["sec_per_frame"]))  # min 5 minutes
        update_config(
            project_dir, 
            {step: {"slurm_params": {
                "time": format_time_from_sec(time_sec),
            }}},
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

    # Generate the slurm scripts
    time.sleep(1)
    config = load_config(project_dir)
    for step in processing_steps:
        slurm_cmd = _make_slurm_cmd(
            **config[step]['slurm_params'], 
            slurm_out_prefix=f"{step}_{current_time}",
        )
        wrap_cmd = _make_wrap_cmd(**config[step]['wrap_params'])
        cmd = slurm_cmd + wrap_cmd
        out_file = join(config[step]["slurm_params"]["step_dir"], f"{step}_batch_{current_time}.sh")
        videos = [v["video_path"] for v in config["VID_INFO"].values()]
        with open(out_file, "w") as f:
            for vid in videos:
                args_str = " ".join([arg for arg in config[step]["func_args"].values()]).format(video_path=vid)
                full_cmd = cmd.format(args=args_str)
                f.write(full_cmd)
                f.write("\n\n")

        print(f"Batch script for step {step} ready at {out_file}")

    return


def _make_slurm_cmd(
    time,
    gpu,
    mem,
    ncpus,
    slurm_out_dir,
    slurm_out_prefix,
    **kwargs,
):
    """Generate the slurm portion of the script."""

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
        f'-o {slurm_out_dir}/slurm_{slurm_out_prefix}_%j.out '
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
    """Generate the wrap portion of the script."""

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

    Arguments:
        project_dir {str} -- path to the project directory
        processing_step {str} -- step of keypoint processing to run

    Keyword Arguments:
        shell_script {str} -- script to run instead of the default (default: {None})
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

    # Make sure jobs aren't already running
    # TODO: allow user to do parameter scans?
    if top_level in config and "jobs_in_progress" in config[top_level]["slurm_params"]:
        jip = config[top_level]["slurm_params"]["jobs_in_progress"]
        if jip:
            print(f"Jobs already running for {top_level}: {jip}")
            return

    # Run the shell script
    print(f"Running script {os.path.basename(shell_script_to_run)}")
    os.system(f'chmod +x {shell_script_to_run}')
    out = os.popen(f'{shell_script_to_run}').read()
    print(out)

    # Update the config with the list of running jobs
    submitted_jobs = re.findall(r"Submitted batch job (\d+)", out)
    update_config(project_dir, {top_level: {"slurm_params": {"jobs_in_progress": submitted_jobs}}})

    return


def cancel_batch(project_dir, processing_step, shell_script=None):
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
    update_config(project_dir, {top_level: {"slurm_params": {"jobs_in_progress": []}}}, verbose=False)

    return