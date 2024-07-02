import multiprocessing
import os
from os.path import join

import av
import click

# see also imports under main()
from multicamera_keypoints.vid_utils import count_frames


def make_config(
    PACKAGE_DIR,
    compression_overwrites_original,
    sec_per_frame=0.06,
    output_name_suffix=None,
    step_dependencies=None,
    compression_kwargs_dict=None,
):
    """Create a default config for the COMPRESSION step.

    Parameters
    ----------
    PACKAGE_DIR : str
        The directory where the package is installed.

    compression_overwrites_original: bool
        Whether or not the compression step should overwrite the original video file.
        This is a required argument because it is important that the user is conscious of the fact that the original video file will be overwritten.
        
    sec_per_frame : float, optional
        The number of seconds per frame for the compression step. The default is 0.06 (for ffmpeg preset of "slow"),
        which is approximately how long it takes using 8 threads and the fmtgray flag.
        
    output_name_suffix : str, optional
        The suffix to add to the output name. The default is None.
        Example: "CMPR_crf21" --> "[session_name].CMPR_crf21.mp4", and the step name 
        will be "COMPRESSION.CMPR_crf21".

    step_dependencies : list, optional
        The list of step names for the dependencies of this step. The default is None, i.e. [].
        These steps will be checked for completion before running this step.

    compression_kwargs_dict : dict
        Additional keyword arguments to pass to the compression step.
        Parameters are:
            - preset : str
                ffmpeg preset for compression. Options are: slow (default), fast, etc.
            - crf : int
                Constant Rate Factor for ffmpeg compression. Default is 21.
            # - original_framerate : int
            #     Original framerate of the video. If not provided, the framerate will be read from the video.
            - nthreads : int
                Number of threads for ffmpeg multi-threading. -1 (default) means use all available. ffmpeg max is 16.
            - fmtgray : bool
                Experimental feature: add grayscale filter before compressing. Speeds things up, reduces file size.

    Returns
    -------
    compression_config : dict
        The configuration for the compression step. 

    step_name : str
        The name of the detection step. (default: "COMPRESSION")
    """

    # Name the step based on output name, if provided
    if output_name_suffix is not None:
        step_name = f"COMPRESSION.{output_name_suffix}"
    else:
        step_name = "COMPRESSION"

    if step_dependencies is None:
        step_dependencies = []

    # Add kwargs to compression kwargs dict
    if output_name_suffix is not None:
        compression_kwargs_dict["output_name_suffix"] = output_name_suffix
    compression_kwargs_dict["delete_original"] = compression_overwrites_original

    compression_config = {
        "slurm_params": {
            "mem": "8GB",
            "gpu": False,
            "sec_per_frame": sec_per_frame,
            "ncpus": 8,
            "jobs_in_progress": {},
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "compression", "compression.py"),
            "conda_env": "dataPy_torch2",  # TODO: make this dynamic?
        },
        "func_args": {"video_path": "{video_path}"},
        "func_kwargs": compression_kwargs_dict,
        "output_info": {"expected_post_comp_max_kb_per_frame": 25, "output_name": output_name_suffix},
        "step_dependencies": step_dependencies,
        "pipeline_info": {
            "processing_level": "video",
        }
    }

    return compression_config, step_name


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output_name_suffix", default=None, help="The suffix to add to the output name. Example: 'CMPR_crf21' --> '[session_name].CMPR_crf21.mp4'.")
@click.option("--preset", default="slow", help="Preset for ffmpeg compression")
@click.option("--crf", default=21, help="CRF for ffmpeg compression")
@click.option("--nthreads", default=-1, help="Number of threads for ffmpeg multi-threading. -1 (default) means use all available. ffmpeg max is 16.")
@click.option("--fmtgray", is_flag=True, help="Experimental feature: add grayscale filter before compressing.")
@click.option("--delete_original", is_flag=True, help='Whether or not to delete the original file. If true, the original file will be replaced with the compressed file. If false, the compressed file will be saved as input_vid.replace(ext, f"_COMPRESSED{ext}").')
@click.option("--force_framerate", default=None, help='If provided, use this instead of calculating the frame rate (speeds up debugging).')
def main(
    video_path,
    output_name_suffix=None,
    preset="slow",
    crf=21,
    nthreads=-1,
    delete_original=False,
    fmtgray=False,
    force_framerate=None,
):
    """
    Compress a video using ffmpeg.

    See discussion here about different ffmpeg presets: https://superuser.com/questions/1556953/why-does-preset-veryfast-in-ffmpeg-generate-the-most-compressed-file-compared
    The takeaway is, if you use a "fast" preset, you might get what looks like the same file size — but the quality will be worse.
    Given that we're optimizing for data integrity and we have plenty of CPU time, the "slow" preset is a good choice.

    Parameters
    ----------
    input_vid : str
        Path to input video.

    output_name_suffix : str
        The suffix to add to the output name. Example: "CMPR_crf21" --> "[session_name].CMPR_crf21.mp4".

    preset : str
        Preset for compression. Options are: slow (TODO: add more options)

    delete_original : bool
        Whether or not to delete the original file. If true and no output vid is provided, the original file will be replaced with the compressed file.

    Returns
    -------
    None

    """
    # Set up
    print(f"Running compression on {video_path} with presest {preset}")

    # Read frame rate from video using pyav
    # # if original_framerate is None:
    #     container = av.open(video_path)
    #     stream = container.streams.video[0]
    #     original_framerate = stream.average_rate
    #     container.close()
    #     print(f"Original framerate detected: {original_framerate} fps")
    # else:
    #     print(f"Using user-provided framerate: {original_framerate} fps")
    #     original_framerate = int(original_framerate)

    if force_framerate is not None:
        original_framerate = float(force_framerate)
        assert isinstance(original_framerate, float), "Forced framerate must be a float."
    else:
        container = av.open(video_path)
        stream = container.streams.video[0]
        original_framerate = float(stream.average_rate)
        container.close()
        print(f"Original framerate detected: {original_framerate} fps")

    # Count num cores available
    if isinstance(nthreads, str):
        nthreads = int(nthreads)
    if nthreads == -1:    
        try:
            # assume we're on slurm
            nthreads = os.getenv('SLURM_CPUS_PER_TASK')
        except KeyError:
            nthreads = multiprocessing.cpu_count()
    else:
        nthreads = int(nthreads)
        assert nthreads > 0, "Number of threads must be greater than 0."


    # Set up output path
    if delete_original is False and output_name_suffix is None:
        output_vid = video_path.replace(".mp4", ".COMPRESSED.mp4")
        replace_original = False
    elif delete_original is False and output_name_suffix is not None:
        output_vid = video_path.replace(".mp4", f".{output_name_suffix}.mp4")
        replace_original = False
    elif delete_original is True and output_name_suffix is not None:
        output_vid = video_path.replace(".mp4", f".{output_name_suffix}.mp4")
        replace_original = True  # TODO: this should be false.
    elif delete_original is True and output_name_suffix is None:
        output_vid = video_path.replace(".mp4", ".tmp.mp4")
        replace_original = True

    output_dir = os.path.dirname(output_vid)
    vid_name = os.path.basename(video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Output video will be saved to: {output_vid}")

    if replace_original:
        print("The original video will be replaced with the compressed video!")

    # Set ffmpeg options
    if fmtgray:
        fmt_filter = "-vf format=gray,format=yuv420p"
    else:
        fmt_filter = ""
    
    ffmpeg_command = f"ffmpeg -y -r {original_framerate} -i {video_path} {fmt_filter} -c:v libx264 -preset {preset} -crf {crf} -threads {nthreads} {output_vid}"

    # Count frames before compression, to ensure they match after compression
    num_frames = count_frames(video_path)
    original_filesize = os.path.getsize(video_path)

    # TODO: check that replace_original is set correctly, seems like orig is getting replaced even when user doesn't want it to be 
    # with --output_name_suffix given and --delete_original not set, works as expected
    # with neither given, works as expected
    
    # import pdb
    # pdb.set_trace()

    # Compress video, and read out the result
    if not os.path.exists(output_vid) or delete_original:
        sys_out = os.popen(ffmpeg_command).read()
        with open(f"{output_dir}/{vid_name}_COMPRESSION_log.txt", "w") as f:
            f.write(sys_out)  # Save the ffmpeg logs
    else:
        print(f"File already exists: {output_vid}")

    # Count frames of output vid, ensure matching
    print("Checking number of frames in new video")
    compressed_num_frames = count_frames(output_vid)
    if num_frames != compressed_num_frames:
        raise ValueError(
            f"Number of frames do not match, original vid has {num_frames} and compressed vid has {compressed_num_frames} frames."
        )
    
    # Check file size
    compressed_filesize = os.path.getsize(output_vid)
    print(f"Original file size: {original_filesize/1e9:0.3f} GB")
    print(f"Compressed file size: {compressed_filesize/1e9:0.3f} GB")
    print(f"Compression ratio: {compressed_filesize/original_filesize:0.3f}")

    # Remove the original file if requested
    if (num_frames == compressed_num_frames) and delete_original:
        print("Removing old video!")
        os.remove(video_path)

    # Replace the original file with the compressed file if requested
    if replace_original:
        print("Replacing original file with compressed file!")
        os.rename(output_vid, video_path)

    print("Done.")


if __name__ == "__main__":
    main()
