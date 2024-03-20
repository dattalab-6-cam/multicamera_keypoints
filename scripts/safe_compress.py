import av
import os
import sys
import numpy as np
import click


def count_frames(file_name):
    if os.path.exists(file_name):
        try:
            with av.open(file_name, "r") as reader:
                return reader.streams.video[0].frames
        except Exception as e:
            print(e)
    else:
        print("File does not exist")


@click.command()
@click.argument("input_vid", type=click.Path(exists=True))
@click.argument("--output_vid", default=None)
@click.option("--original_framerate", default="120", help="Original framerate of the video")
@click.option("--preset", default="slow", help="Preset for compression")
@click.option("--delete_original", is_flag=True, help="Whether or not to delete the original file. If true and no output vid is provided, the original file will be replaced with the compressed file. If false and no output vid is provided, the compressed file will be saved as input_vid.replace(ext, f\"_COMPRESSED{ext}\").")
def main(input_vid, output_vid=None, original_framerate="120", preset="slow", delete_original=False):
    """
    Compress a video using ffmpeg.

    See discussion here about different ffmpeg presets: https://superuser.com/questions/1556953/why-does-preset-veryfast-in-ffmpeg-generate-the-most-compressed-file-compared
    The takeaway is, if you use a "fast" preset, you might get what looks like the same file size — but the quality will be worse.
    Given that we're optimizing for data integrity and we have plenty of CPU time, the "slow" preset is a good choice.

    Parameters
    ----------
    input_vid : str
        Path to input video.

    output_vid : str
        Path to output video. 
        If not provided and delete_original is False, the compressed video will be saved as input_vid.replace(ext, f"_COMPRESSED{ext}").
        If not provided and delete_original is True, the compressed video will replace the original video.
        If provided, the compressed video will be saved to the provided path, and the original deleted or not depending on the delete_original flag.

    original_framerate : int
        Original framerate of the video.

    preset : str
        Preset for compression. Options are: slow (TODO: add more options)

    delete_original : bool
        Whether or not to delete the original file. If true and no output vid is provided, the original file will be replaced with the compressed file.

    Returns
    -------
    None

    """
    # Set up 
    print(f"Running compression on {input_vid} with presest {preset}")
    original_framerate = int(original_framerate)

    # Set up output path
    if output_vid is None and delete_original is False:
        output_vid = input_vid.replace(".mp4", "_COMPRESSED.mp4")
        replace_original = False
    elif output_vid is None and delete_original is True:
        output_vid = input_vid.replace(".mp4", "_COMPRESSED.mp4")
        replace_original = True

    output_dir = os.path.dirname(output_vid)
    vid_name = os.path.basename(input_vid)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set ffmpeg options
    if preset == "slow":
        ffmpeg_command = f"ffmpeg -y -r {original_framerate} -i {input_vid} -c:v libx264 -preset slow -crf 23 {output_vid}"

    # Count frames
    num_frames = count_frames(input_vid)

    # Compress video, and read out the result
    if not os.path.exists(output_vid):
        sys_out = os.popen(ffmpeg_command).read()
        with open(f"{output_dir}/{vid_name}_COMPRESSION_log.txt", "w") as f:
            f.write(sys_out)  # Save the ffmpeg logs
    else:
        print(f"File already exists: {output_vid}")

    # Count frames of output vid, ensure matching
    print("Checking number of frames in new video")
    compressed_num_frames = count_frames(output_vid)
    if num_frames != compressed_num_frames:
        raise ValueError(f"Number of frames do not match, original vid has {num_frames} and compressed vid has {compressed_num_frames} frames.")

    # Remove the original file if requested
    if delete_original:
        print("Removing old video!")
        os.remove(input_vid)

    # Replace the original file with the compressed file if requested
    if replace_original:
        print("Replacing original file with compressed file!")
        os.rename(output_vid, input_vid)
    
    print("Done.")

if __name__ == "__main__":
    main()