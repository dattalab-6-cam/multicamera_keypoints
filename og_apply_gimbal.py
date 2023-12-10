import jax
import sys
import glob
import gimbal
import joblib
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import joblib, json, os, h5py
import imageio, cv2
import tqdm
import matplotlib.pyplot as plt
import multicam_calibration as mcc
import keypoint_moseq as kpms
from scipy.ndimage import median_filter
jax.config.update("jax_enable_x64", False)
import yaml, sys
from keypoint_sort.util import build_node_hierarchy
from keypoint_moseq.util import get_edges

def generate_gimbal_params(
    camera_matrices, 
    fitted_params,
    obs_outlier_variance,
    pos_dt_variance,
    num_leapfrog_steps=5,
    step_size = 1e-1,
    dtype='float32'):
    
    num_cameras, num_joints = fitted_params['obs_inlier_variance'].shape

    params = {
        'obs_outlier_probability' : jnp.zeros((num_cameras, num_joints), dtype),
        'obs_outlier_location': jnp.zeros((num_cameras, num_joints,2), dtype),
        'obs_outlier_variance': jnp.ones((num_cameras, num_joints), dtype)*obs_outlier_variance,
        'obs_inlier_location': jnp.zeros((num_cameras, num_joints,2), dtype),
        'obs_inlier_variance': jnp.array(fitted_params['obs_inlier_variance'], dtype),
        'camera_matrices': jnp.array(camera_matrices, dtype),
        'pos_radius': jnp.array(fitted_params['radii'], dtype),
        'pos_radial_variance': jnp.array([1e8,*fitted_params['radii_std'][1:]**2], dtype),
        'parents': jnp.array(fitted_params['parents']),
        'pos_dt_variance': jnp.ones(num_joints, dtype)*pos_dt_variance,
        'state_probability': jnp.array(fitted_params['pis'], dtype),
        'state_directions': jnp.array(fitted_params['mus'], dtype),
        'state_concentrations': jnp.array(fitted_params['kappas'], dtype),
        'crf_keypoints': fitted_params['indices_egocentric'][::-1],
        'crf_abscissa': jnp.zeros(3, dtype).at[0].set(1),
        'crf_normal': jnp.zeros(3, dtype).at[2].set(1),
        'crf_axes': jnp.eye(3).astype(dtype),
        'state_transition_count': jnp.ones(len(fitted_params['pis']), dtype),
        'step_size': step_size, 'num_leapfrog_steps':num_leapfrog_steps,
    }
    return gimbal.mcmc3d_full.initialize_parameters(params)


def generate_initial_positions(positions):
    init_positions = np.zeros_like(positions)
    for k in range(positions.shape[1]):
        ix = np.nonzero(~np.isnan(positions[:,k,0]))[0]
        for i in range(positions.shape[2]):
            init_positions[:,k,i] = np.interp(
                np.arange(positions.shape[0]),
                ix, positions[:,k,i][ix])
    return init_positions
        
    
def generate_outlier_probs(confidence,     
    outlier_prob_bounds=[1e-3,1-1e-6],
    conf_sigmoid_center=0.3,
    conf_sigmoid_gain=20):
    
    outlier_p = jax.nn.sigmoid((conf_sigmoid_center-confidence)*conf_sigmoid_gain)
    return jnp.clip(outlier_p, *outlier_prob_bounds)





bodyparts = ['tail_tip',
             'tail_base',
             'spine_low',
             'spine_mid',
             'spine_high',
             'left_ear',
             'right_ear',
             'forehead',
             'nose_tip',
             'left_hind_paw_front',
             'left_hind_paw_back',
             'right_hind_paw_front',
             'right_hind_paw_back',
             'left_fore_paw',
             'right_fore_paw']

skeleton = [
    ['tail_base', 'spine_low'],
    ['spine_low', 'spine_mid'],
    ['spine_mid', 'spine_high'],
    ['spine_high', 'left_ear'],
    ['spine_high', 'right_ear'],
    ['spine_high', 'forehead'],
    ['forehead', 'nose_tip'],
    ['left_hind_paw_back', 'left_hind_paw_front'],
    ['spine_low', 'left_hind_paw_back'],
    ['right_hind_paw_back', 'right_hind_paw_front'],
    ['spine_low', 'right_hind_paw_back'],
    ['spine_high', 'left_fore_paw'],
    ['spine_high', 'right_fore_paw']
]


use_bodyparts = bodyparts[1:]
use_bodyparts_ix = np.array([bodyparts.index(bp) for bp in use_bodyparts])
edges = np.array(get_edges(use_bodyparts, skeleton))
node_order, parents = build_node_hierarchy(use_bodyparts, skeleton, 'spine_low')
edges = np.argsort(node_order)[edges]
fitted_params = joblib.load('/n/groups/datta/Jonah/20231121_6cam_sniff/kp_detection/gimbal_params.p')


#calib_path = '/n/groups/datta/Jonah/kpms_reviews_6cam_thermistor/raw_data/calibration/data/20230904_calibration/camera_params.h5'
#vid_dir = '/n/groups/datta/Jonah/kpms_reviews_6cam_thermistor/raw_data/J01602/20230904_J01602'

calib_path = sys.argv[1]
vid_dir = sys.argv[2]

print(f"Running gimbal on {vid_dir}")
print(jax.default_backend())

all_extrinsics, all_intrinsics, camera_names = mcc.load_calibration(calib_path, 'gimbal')

observations = []
confidence = []
for i,c in tqdm.tqdm(enumerate(camera_names)):
    with h5py.File(f'{vid_dir}/{c}.keypoints.h5','r') as h5:
        uvs = h5['uv'][()][:,use_bodyparts_ix][:,node_order]
        uvs = mcc.undistort_points(uvs, *all_intrinsics[i])
        observations.append(uvs)
        confidence.append(h5['conf'][()][:,use_bodyparts_ix][:,node_order])
        
observations = np.stack(observations,axis=1)
confidence = np.stack(confidence,axis=1)
triangulation_positions = np.load(f'{vid_dir}/robust_triangulation.npy')[:,use_bodyparts_ix][:,node_order]


camera_matrices = np.array([
    mcc.get_projection_matrix(extrinsics, intrinsics)
    for extrinsics, intrinsics in zip(all_extrinsics, all_intrinsics)])

params = generate_gimbal_params(camera_matrices, fitted_params, 1e6, 1, step_size=.2)

# initialize positions
print("Generating init positions")
init_positions = generate_initial_positions(triangulation_positions)

# calculate probabilies
print("Generating probs")
outlier_prob = generate_outlier_probs(
    confidence, outlier_prob_bounds=[1e-6, 1-1e-6], 
    conf_sigmoid_center=0.5,
    conf_sigmoid_gain=20)

# initialize gimbal state
print("Init gimbal")
init_positions = jnp.array(init_positions, 'float32')
observations = jnp.array(observations, 'float32')
outlier_prob = jnp.array(outlier_prob, 'float32')
samples = gimbal.mcmc3d_full.initialize(
    jr.PRNGKey(0), params, observations, outlier_prob, init_positions)

num_iterations = 5000
positions_sum = np.zeros_like(init_positions)
tot = 0

print("Fitting gimbal")
log_likelihood = []
for itr in tqdm.tqdm(range(num_iterations)):
    samples = gimbal.mcmc3d_full.step(jr.PRNGKey(itr), params, observations, outlier_prob, samples)
    log_likelihood.append(samples['log_probability'].item())

    if itr % 5 == 0 and itr > 4000:
        positions_sum += np.array(samples['positions'])
        tot += 1
    
    if itr % 10 == 0: 
        tqdm.tqdm.write(f"Iter {itr}, ll: {log_likelihood[-1]}")
        
positions_mean = positions_sum / tot
np.save(f'{vid_dir}/gimbal.npy', positions_mean)
np.save(vid_dir.split('/')[-1]+'.ll.npy',log_likelihood)
