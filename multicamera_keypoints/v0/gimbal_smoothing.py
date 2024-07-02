import os
from os.path import join

import click
import joblib
import multicam_calibration as mcc
import numpy as np
import tqdm.auto as tqdm
from o2_utils.selectors import find_files_from_pattern
# see also imports under main()

def make_config(
    PACKAGE_DIR,
    gimbal_params_path,
    sec_per_frame=0.09,
    output_name_suffix=None,
    step_dependencies=None,
):
    """Create a default config for the GIMBAL step.

    Parameters
    ----------
    PACKAGE_DIR : str
        The directory where the package is installed.

    sec_per_frame : float, optional
        The number of seconds per frame for the gimbal step. The default is 0.09.

    output_name_suffix : str, optional
        The suffix to add to the output name. The default is None.
        Example: "v2" --> "gimbal.v2.npy", and the step name
        will be "GIMBAL.v2".

    step_dependencies : list, optional
        The list of step names for the dependencies of this step. The default is ["TRIANGULATION"].
        These steps will be checked for completion before running this step.

    Returns
    -------
    gimbal_config : dict
        The configuration for the gimbal step.

    step_name : str
        The name of the smoothing step. (default: "GIMBAL")
    """
    if output_name_suffix is not None:
        output_name = f"gimbal.{output_name_suffix}.npy"
        step_name = f"GIMBAL.{output_name_suffix}"
    else:
        output_name = "gimbal.npy"
        step_name = "GIMBAL"

    if step_dependencies is None:
        step_dependencies = ["TRIANGULATION"]

    gimbal_config = {
        "slurm_params": {
            "mem": "8GB",
            "gpu": True,
            "sec_per_frame": sec_per_frame,
            "ncpus": 1,
            "jobs_in_progress": {},
        },
        "wrap_params": {
            "func_path": join(PACKAGE_DIR, "v0", "gimbal_smoothing.py"),
            "conda_env": "dataPy_KPMS_GIMBAL",  # TODO: make this dynamic
            "modules": ["gcc/9.2.0", "ffmpeg", "cuda/11.7"],
        },
        "func_args": {  # NB: these args **must** be in the right order here.
            "vid_dir": "{video_dir}",
            "calib_file": "{calib_file}",
            "gimbal_params_file": gimbal_params_path,
        },
        "output_info": {
            "output_name": output_name,
        },
        "step_dependencies": step_dependencies,
        "pipeline_info": {
            "processing_level": "session",
        },
    }

    return gimbal_config, step_name


def generate_gimbal_params(
    camera_matrices,
    fitted_params,
    obs_outlier_variance,
    pos_dt_variance,
    num_leapfrog_steps=5,
    step_size=1e-1,
    dtype="float32",
):
    num_cameras, num_joints = fitted_params["obs_inlier_variance"].shape

    params = {
        "obs_outlier_probability": jnp.zeros((num_cameras, num_joints), dtype),
        "obs_outlier_location": jnp.zeros((num_cameras, num_joints, 2), dtype),
        "obs_outlier_variance": jnp.ones((num_cameras, num_joints), dtype)
        * obs_outlier_variance,
        "obs_inlier_location": jnp.zeros((num_cameras, num_joints, 2), dtype),
        "obs_inlier_variance": jnp.array(fitted_params["obs_inlier_variance"], dtype),
        "camera_matrices": jnp.array(camera_matrices, dtype),
        "pos_radius": jnp.array(fitted_params["radii"], dtype),
        "pos_radial_variance": jnp.array(
            [1e8, *fitted_params["radii_std"][1:] ** 2], dtype
        ),
        "parents": jnp.array(fitted_params["parents"]),
        "pos_dt_variance": jnp.ones(num_joints, dtype) * pos_dt_variance,
        "state_probability": jnp.array(fitted_params["pis"], dtype),
        "state_directions": jnp.array(fitted_params["mus"], dtype),
        "state_concentrations": jnp.array(fitted_params["kappas"], dtype),
        "crf_keypoints": fitted_params["indices_egocentric"][::-1],
        "crf_abscissa": jnp.zeros(3, dtype).at[0].set(1),
        "crf_normal": jnp.zeros(3, dtype).at[2].set(1),
        "crf_axes": jnp.eye(3).astype(dtype),
        "state_transition_count": jnp.ones(len(fitted_params["pis"]), dtype),
        "step_size": step_size,
        "num_leapfrog_steps": num_leapfrog_steps,
    }
    return gimbal.mcmc3d_full.initialize_parameters(params)


def generate_initial_positions(positions):
    init_positions = np.zeros_like(positions)
    for k in range(positions.shape[1]):
        ix = np.nonzero(~np.isnan(positions[:, k, 0]))[0]
        for i in range(positions.shape[2]):
            init_positions[:, k, i] = np.interp(
                np.arange(positions.shape[0]), ix, positions[:, k, i][ix]
            )
    return init_positions


def generate_outlier_probs(
    confidence,
    outlier_prob_bounds=[1e-3, 1 - 1e-6],
    conf_sigmoid_center=0.3,
    conf_sigmoid_gain=20,
):
    outlier_p = jax.nn.sigmoid((conf_sigmoid_center - confidence) * conf_sigmoid_gain)
    return jnp.clip(outlier_p, *outlier_prob_bounds)


@click.command()
@click.argument("vid_dir")
@click.argument("calib_file")
@click.argument("gimbal_params_file")
@click.option(
    "--checkpoint_niters",
    default=500,
    help="Number of iterations between saving checkpoints",
)
@click.option("--output_name", default="gimbal.npy", help="Suffix to add to output name")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(
    vid_dir,
    calib_file,
    gimbal_params_file, 
    checkpoint_niters=500, 
    output_name=None,
    overwrite=False, 
):
    
    print("Running gimbal smoothing on ", vid_dir)

    # Hard-coded KP params -- fix this
    bodyparts = [
        "tail_tip",
        "tail_base",
        "spine_low",
        "spine_mid",
        "spine_high",
        "left_ear",
        "right_ear",
        "forehead",
        "nose_tip",
        "left_hind_paw_front",
        "left_hind_paw_back",
        "right_hind_paw_front",
        "right_hind_paw_back",
        "left_fore_paw",
        "right_fore_paw",
    ]

    skeleton = [
        ["tail_base", "spine_low"],
        ["spine_low", "spine_mid"],
        ["spine_mid", "spine_high"],
        ["spine_high", "left_ear"],
        ["spine_high", "right_ear"],
        ["spine_high", "forehead"],
        ["forehead", "nose_tip"],
        ["left_hind_paw_back", "left_hind_paw_front"],
        ["spine_low", "left_hind_paw_back"],
        ["right_hind_paw_back", "right_hind_paw_front"],
        ["spine_low", "right_hind_paw_back"],
        ["spine_high", "left_fore_paw"],
        ["spine_high", "right_fore_paw"],
    ]

    # Check output name
    assert output_name.endswith(".npy"), "Output name must end with .npy"

    # Check that the output file doesn't already exist
    triang_file = find_files_from_pattern(vid_dir, "*.robust_triangulation.npy")
    session_name = os.path.basename(triang_file).split(".")[0]
    gimbal_out_file = os.path.join(vid_dir, f"{session_name}.{output_name}")
    ll_out_file = os.path.join(vid_dir, f"{session_name}.ll.npy")
    if os.path.exists(gimbal_out_file) and not overwrite:
        print(f"Output file {gimbal_out_file} already exists. Exiting.")
        return

    # import pdb
    # pdb.set_trace()

    # # Check that a checkpoint file doesn't already exist
    # if os.path.exists(gimbal_out_file.replace("gimbal.npy", "gimbal_CHKPT.npy")) and not overwrite:
    #     print(f"Loading checkpoint file {gimbal_out_file.replace('gimbal.npy', 'gimbal_CHKPT.npy')}")
    #     positions_sum = np.load(gimbal_out_file.replace("gimbal.npy", "gimbal_CHKPT.npy"))

    use_bodyparts = bodyparts[1:]
    use_bodyparts_ix = np.array([bodyparts.index(bp) for bp in use_bodyparts])
    edges = np.array(get_edges(use_bodyparts, skeleton))
    node_order, parents = build_node_hierarchy(use_bodyparts, skeleton, "spine_low")
    edges = np.argsort(node_order)[edges]
    fitted_params = joblib.load(gimbal_params_file)

    all_extrinsics, all_intrinsics, camera_names = mcc.load_calibration(
        calib_file, "gimbal"
    )

    observations = []
    confidence = []
    for i, c in tqdm.tqdm(enumerate(camera_names)):
        h5_file = find_files_from_pattern(vid_dir, f"*{c}*.keypoints.h5")
        with h5py.File(h5_file) as h5:
            uvs = h5["uv"][()][:, use_bodyparts_ix][:, node_order]
            uvs = mcc.undistort_points(uvs, *all_intrinsics[i])
            observations.append(uvs)
            confidence.append(h5["conf"][()][:, use_bodyparts_ix][:, node_order])

    observations = np.stack(observations, axis=1)
    confidence = np.stack(confidence, axis=1)
    triangulation_positions = np.load(triang_file)[:, use_bodyparts_ix][:, node_order]

    camera_matrices = np.array(
        [
            mcc.get_projection_matrix(extrinsics, intrinsics)
            for extrinsics, intrinsics in zip(all_extrinsics, all_intrinsics)
        ]
    )

    params = generate_gimbal_params(
        camera_matrices, fitted_params, 1e6, 1, step_size=0.2
    )

    # initialize positions
    init_positions = generate_initial_positions(triangulation_positions)

    # calculate probabilies
    outlier_prob = generate_outlier_probs(
        confidence,
        outlier_prob_bounds=[1e-6, 1 - 1e-6],
        conf_sigmoid_center=0.5,
        conf_sigmoid_gain=20,
    )

    # initialize gimbal state
    init_positions = jnp.array(init_positions, "float32")
    observations = jnp.array(observations, "float32")
    outlier_prob = jnp.array(outlier_prob, "float32")
    samples = gimbal.mcmc3d_full.initialize(
        jr.PRNGKey(0), params, observations, outlier_prob, init_positions
    )

    num_iterations = 5000
    positions_sum = np.zeros_like(init_positions)
    tot = 0

    log_likelihood = []
    for itr in range(num_iterations):
        samples = gimbal.mcmc3d_full.step(
            jr.PRNGKey(itr), params, observations, outlier_prob, samples
        )
        log_likelihood.append(samples["log_probability"].item())

        if itr % 5 == 0 and itr > 4000:
            positions_sum += np.array(samples["positions"])
            tot += 1

        if itr % 10 == 0:
            print(itr, log_likelihood[-1], flush=True)

        # if itr % 500 == 0:
        #     np.save(gimbal_out_file.replace("gimbal.npy", f"gimbal_CHKPT_{itr}.npy"), positions_sum / tot)
        #     np.save(ll_out_file.replace("ll.npy", "ll_CHKPT_{itr}.npy"), log_likelihood)

    positions_mean = positions_sum / tot
    np.save(gimbal_out_file, positions_mean)
    np.save(ll_out_file, log_likelihood)

    print(f"Saved gimbal positions to {gimbal_out_file}")

    return


if __name__ == "__main__":
    from keypoint_moseq.util import get_edges
    from keypoint_sort.util import build_node_hierarchy
    import gimbal
    import h5py
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    jax.config.update("jax_enable_x64", False)
    
    main()
