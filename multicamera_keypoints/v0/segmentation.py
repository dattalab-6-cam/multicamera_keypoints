import os
from os.path import join

import click
import cv2
import numpy as np
import tqdm
from o2_utils.selectors import find_files_from_pattern
from vidio.read import OpenCVReader
# see also imports under main()

def make_config(
    PACKAGE_DIR,
    weights_path,
    sec_per_frame=0.021,
    output_name_suffix=None,
    step_dependencies=None,
):
    """Create a default config for the CENTERNET step.

    Parameters
    ----------
    PACKAGE_DIR : str
        The directory where the package is installed.

    weights_path : str
        The path to the weights file for the centernet model.
        
    sec_per_frame : float, optional
        The number of seconds per frame for the centernet step. The default is 0.021.
        
    output_name_suffix : str, optional
        The suffix to add to the output name. The default is None.
        Example: "v2" --> "centroid.v2.npy", and the step name 
        will be "CENTERNET.v2".

    step_dependencies : list, optional
        The list of step names for the dependencies of this step. The default is None.
        These steps will be checked for completion before running this step.

    Returns
    -------
    centernet_config : dict
        The configuration for the centernet step. 

    step_name : str
        The name of the segmentation step. (default: "CENTERNET")
    """
    if output_name_suffix is not None:
        output_name = f"centroid.{output_name_suffix}.npy"
        step_name = f"CENTERNET.{output_name_suffix}"
    else:
        output_name = "centroid.npy"
        step_name = "CENTERNET"

    centernet_config = {
        "slurm_params": {
            "mem": "4GB",
            "gpu": True,
            "sec_per_frame": sec_per_frame,  # 75 min/job x 60 / (30*60*120 frames/job) = 0.021 sec/frame
            "ncpus": 2,
            "jobs_in_progress": {},
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "segmentation.py"),
            "conda_env": "dataPy_torch2",  # TODO: make this dynamic?
        },
        "func_args": {  # NB: these args **must** be in the right order here.
            "video_path": "{video_path}",
            "weights_path": find_files_from_pattern(
                weights_path, "centernet.pth"
            ),  # TODO: get these func args from a more reasonable location, ie the function should specify what its args are
        },
        "output_info": {
            "output_name": output_name,
        },
    }

    if step_dependencies is not None:
        centernet_config["step_dependencies"] = step_dependencies

    return centernet_config, step_name


def load_model(weights_path):
    model = ConvNet(initial_channels=32)
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    return model


def apply_model(model, vid_path):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    reader = OpenCVReader(vid_path)
    nframes = len(reader)

    uvs = []
    for i in tqdm.trange(nframes):
        with torch.no_grad():
            im = clahe.apply(reader[i][::2, ::2, 0])
            x = im[None, None].astype(np.float32) / 255
            y = model(torch.Tensor(x).to("cuda")).detach().cpu().numpy()[0, 0]
            y = cv2.GaussianBlur(y, (7, 7), 2)
            uvs.append([np.argmax(y.max(0)), np.argmax(y.max(1))])
    uvs = np.array(uvs) * 32
    return uvs


@click.command()
@click.argument("vid_path")
@click.argument("weights_path")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(vid_path, weights_path, overwrite=False):
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from mck.v0.convnet import ConvNet
    
    print(f"Running centernet on {vid_path}...")
    fullfile, ext = os.path.splitext(vid_path)
    output_name = "centroid"
    out_file = fullfile + "." + output_name + ".npy"
    if os.path.exists(out_file) and not overwrite:
        print(f"Output file {out_file} exists, skipping...")
        return

    # Load the model
    print(f"Loading weights from {weights_path}")
    model = load_model(weights_path)

    # Apply the model
    print(f"Applying model to {vid_path}")
    model = model.eval().to("cuda")
    uvs = apply_model(model, vid_path)

    # Save the results
    np.save(out_file, uvs)
    print(f"Results saved as {out_file}!")

    return


if __name__ == "__main__":
    main()
