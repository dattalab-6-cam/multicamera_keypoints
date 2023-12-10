import tqdm
import numpy as np
import h5py
import os
import sys
import cv2
from vidio.read import OpenCVReader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, initial_channels=32):
        super(ConvNet, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(1, initial_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(initial_channels, initial_channels*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(initial_channels*2)
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(initial_channels*2, initial_channels*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(initial_channels*4)
        
        # Fourth Convolutional Layer
        self.conv4 = nn.Conv2d(initial_channels*4, initial_channels*4, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(initial_channels*4)
        
        # MaxPooling Layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final Convolutional Layer to generate 1-channel output
        self.conv_final = nn.Conv2d(initial_channels*4, 1, kernel_size=1, stride=1)
        
    def forward(self, x):
        # First Conv + ReLU + MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Second Conv + ReLU + MaxPool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Third Conv + ReLU + MaxPool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Fourth Conv + ReLU + MaxPool
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Final Conv
        x = self.conv_final(x)
        x = F.sigmoid(x)
        return x


def load_model(weights_path):
    model = ConvNet(initial_channels=32)
    if torch.cuda.is_available():
        weights = torch.load(weights_path)
    else:
        weights = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights)
    return model


def apply_model(vid_path):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    reader = OpenCVReader(vid_path)
    nframes = len(reader)

    uvs = []
    for i in tqdm.trange(nframes):
        with torch.no_grad():
            im = clahe.apply(reader[i][::2,::2,0])
            x = im[None,None].astype(np.float32)/255
            y = model(torch.Tensor(x).to('cuda')).detach().cpu().numpy()[0,0]
            y = cv2.GaussianBlur(y, (7,7), 2)
            uvs.append([np.argmax(y.max(0)), np.argmax(y.max(1))])
    uvs = np.array(uvs)*32
    return uvs

# TODO: add overwrite args
def save_array(array, vid_path, name):
    fullfile, ext = os.path.splitext(vid_path)
    out_file = fullfile + '.' + name + '.npy'
    if not os.path.exists(out_file):
        np.save(out_file, array)
    

def main(vid_path, weights_path):
    model = load_model(weights_path)
    model = model.eval().to('cuda')
    uvs = apply_model(vid_path)
    save_array(uvs, vid_path, 'centroid')

if __name__ == "__main__":
    vid_path = sys.argv[1]
    weights_path = sys.argv[2]
    main(vid_path, weights_path)