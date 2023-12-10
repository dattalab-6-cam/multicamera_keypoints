import torch, tqdm
from hrnet import HRNet
from vidio.read import OpenCVReader
import numpy as np
import h5py
import os, sys
import cv2
# import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def parse_heatmap(heatmap, downsample=4):
    B, C, H, W = heatmap.shape
    flat_heatmap = heatmap.reshape((B, C, -1))
    maxima = torch.argmax(flat_heatmap, dim=-1)
    
    u = maxima % W
    v = torch.div(maxima, W, rounding_mode='floor')
    uv = downsample * torch.stack((u, v), dim=-1)
    
    confidence = torch.gather(flat_heatmap, -1, maxima[..., None])[..., 0]
    confidence = torch.clip(confidence, 0, 1)

    uv = to_numpy(uv).astype(np.int32)
    confidence = to_numpy(confidence).astype(np.float32)
    return uv, confidence

def load_model(weights_path, device='cuda'):
    state_dict = torch.load(weights_path)
    nof_joints = state_dict['final_layer.weight'].shape[0]
    model = HRNet(nof_joints=nof_joints)
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    return model, nof_joints


def crop_image(image, centroid, crop_size):
    """Crop an image around a centroid.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to crop.

    centroid: tuple of int
        (x,y) coordinates of the centroid.

    crop_size: int or tuple(int,int)
        Size of the crop around the centroid. Either a single int for a square
        crop, or a tuple of ints (w,h) for a rectangular crop.

    Returns
    -------
    image: ndarray of shape (crop_size, crop_size, 3)
        Cropped image.
    """
    if isinstance(crop_size, tuple):
        w, h = crop_size
    else:
        w, h = crop_size, crop_size
    x, y = int(centroid[0]), int(centroid[1])

    x_min = max(0, x - w // 2)
    y_min = max(0, y - h // 2)
    x_max = min(image.shape[1], x + w // 2)
    y_max = min(image.shape[0], y + h // 2)

    cropped = image[y_min:y_max, x_min:x_max]
    padded = np.zeros((h, w, *image.shape[2:]), dtype=image.dtype)
    pad_x = max(w // 2 - x, 0)
    pad_y = max(h // 2 - y, 0)
    padded[
        pad_y : pad_y + cropped.shape[0], pad_x : pad_x + cropped.shape[1]
    ] = cropped
    return padded


def main(vid_path):
    camera = vid_path.split('/')[-1].split('.')[0]
    if camera=='bottom': 
        weights_path = '/n/groups/datta/Jonah/kpms_reviews_6cam_thermistor/20230928_avi_compression/train_hrnet-v2/weights/hrnet_bottom.pth'
    elif camera=='top':
        weights_path = '/n/groups/datta/Jonah/kpms_reviews_6cam_thermistor/20230928_avi_compression/train_hrnet-v2/weights/hrnet_top.pth'
    else:
        weights_path = '/n/groups/datta/Jonah/kpms_reviews_6cam_thermistor/20230928_avi_compression/train_hrnet-v2/weights/hrnet_side.pth'

    model, nof_joints = load_model(weights_path)
    model = model.eval().to('cuda')

    fullfile, ext = os.path.splitext(vid_path)
    save_path = fullfile + '.keypoints.h5'

#     if os.path.exists(save_path):
#         print(f'File {save_path} exists, exiting!')
#         return
     
    centroid = median_filter(np.load(fullfile + '.centroid.npy'), (11,1)).astype(int)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))

    reader = OpenCVReader(vid_path)
    nframes = len(reader)
    with h5py.File(save_path, 'w') as h5f:
        h5f.create_dataset('uv', shape=(nframes, nof_joints, 2))
        h5f.create_dataset('conf', shape=(nframes, nof_joints))
        for i, im in tqdm.tqdm(enumerate(reader), total=nframes):
            with torch.no_grad():
                im = crop_image(im, centroid[i], 512)
                im = clahe.apply(im[:,:,0])
                x = im[None,None].astype(np.float32)/255
                y_pred = model(torch.Tensor(x).to('cuda'))
                uv, conf = parse_heatmap(y_pred, downsample=2)
            uv = uv[0] + centroid[i][None,None]-256
            h5f['uv'][i] = uv
            h5f['conf'][i] = conf[0]

if __name__ == "__main__":
    vid_path = sys.argv[1]
    main(vid_path)