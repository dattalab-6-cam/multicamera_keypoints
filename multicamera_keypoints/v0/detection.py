import os
from os.path import join

import click
import cv2
import h5py
import numpy as np
import tqdm
from o2_utils.selectors import find_files_from_pattern
from scipy.ndimage import median_filter
from vidio.read import OpenCVReader
# see also imports under main()

from multicamera_keypoints.v0.hrnet import HRNet
from multicamera_keypoints.vid_utils import crop_image


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

    weights_path : str
        The path to the weights file for the hrnet model.
        
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
    if output_name_suffix is not None:
        output_name = f"keypoints.{output_name_suffix}.h5"
        step_name = f"HRNET.{output_name_suffix}"
    else:
        output_name = "keypoints.h5"
        step_name = "HRNET"

    if step_dependencies is None:
        step_dependencies = ["CENTERNET"]

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


def load_model(weights_path):
    state_dict = torch.load(weights_path)
    nof_joints = state_dict["final_layer.weight"].shape[0]
    model = HRNet(nof_joints=nof_joints)
    model.load_state_dict(state_dict)
    return model, nof_joints


def apply_model(vid_path, model, nof_joints):
    fullfile, ext = os.path.splitext(vid_path)
    centroid = median_filter(np.load(fullfile + ".centroid.npy"), (11, 1)).astype(int)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    reader = OpenCVReader(vid_path)
    nframes = len(reader)
    all_uvs = np.zeros((nframes, nof_joints, 2))
    all_confs = np.zeros((nframes, nof_joints))
    for i, im in tqdm.tqdm(enumerate(reader), total=nframes):
        with torch.no_grad():
            im = crop_image(im, centroid[i], 512)
            im = clahe.apply(im[:, :, 0])
            x = im[None, None].astype(np.float32) / 255
            y_pred = model(torch.Tensor(x).to("cuda"))
            uv, conf = parse_heatmap(y_pred, downsample=2)
        uv = uv[0] + centroid[i][None, None] - 256
        all_uvs[i, ...] = uv
        all_confs[i, :] = conf[0]

    return all_uvs, all_confs


def save_arrays_as_h5(save_path, array_dict):
    with h5py.File(save_path, "w") as h5f:
        for key, arr in array_dict.items():
            h5f.create_dataset(key, shape=arr.shape)
            h5f[key][:] = arr  # if you just write h5f[key] = arr, it will try to create the dataset and fail b/c it already exists.
    return


@click.command()
@click.argument("vid_path")
@click.argument("weights_dir")
@click.option("--output_name", default="keypoints.h5", help="name of output file")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(vid_path, weights_dir, output_name="keypoints.h5", overwrite=False):

    print(f"Detecting keypoints on {vid_path}...")
    assert output_name.split(".")[-1] == "h5", "Output name for keypoint detection must end in .h5"

    camera = vid_path.split("/")[-1].split(".")[1]
    if "bottom" in camera:
        weights_path = find_files_from_pattern(weights_dir, "hrnet_bottom.pth")
    elif "top" in camera:
        weights_path = find_files_from_pattern(weights_dir, "hrnet_top.pth")
    elif "side" in camera:
        weights_path = find_files_from_pattern(weights_dir, "hrnet_side.pth")
    else:
        raise ValueError(f"Unexpected camera name {camera}")

    fullfile, _ = os.path.splitext(vid_path)
    # output_name = "keypoints.h5"
    save_path = fullfile + "." + output_name

    if os.path.exists(save_path) and not overwrite:
        print(f'Output file {save_path} exists, exiting!')
        return
    else:
        print(f"Results will be saved to {save_path}")

    model, nof_joints = load_model(weights_path)
    model = model.eval().to("cuda")
    all_uvs, all_confs = apply_model(vid_path, model, nof_joints)
    array_dict = dict(uv=all_uvs, conf=all_confs)
    save_arrays_as_h5(save_path, array_dict)

    print("Done!")


if __name__ == "__main__":
    import torch
    main()
