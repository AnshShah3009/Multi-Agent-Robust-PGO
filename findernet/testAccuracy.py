import open3d as o3d 
import numpy as np 
import copy
import torch
import math
import argparse
import os
import pandas as pd
import csv
import torch.optim as optim
import matplotlib.pyplot as plt 
from torchvision import transforms
# import DataParser as DP
import torch.nn as nn
from natsort import natsorted
from math import log2
import sys
# import STModel as ST
from tqdm import tqdm
from PIL import Image
import PIL
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import time
import json

from scipy.spatial.transform import Rotation

from myModels import *
from generateDEM import *

def readImg(paths):
    imgs = []
    for i in range(len(paths)):
        img = np.asarray(Image.open(paths[i])).astype(np.float32)
        # img = np.load(paths[i]).astype(np.float32)
        img = cv2.resize(img, (500,500), interpolation=cv2.INTER_NEAREST)

        # imgs.append(img.astype(np.uint8))
        imgs.append(img)
    
    imgs = torch.tensor(np.array(imgs), dtype=torch.float).to(device)
    return imgs


# from index i to j
def comparePose(poses, i, j):
    r1 = Rotation.from_quat(poses[i, 3:])
    r2 = Rotation.from_quat(poses[j, 3:])

    # Calculate the relative rotation from r1 to r2
    relative_rotation = r1.inv() * r2

    # Get Euler angles in radians (Roll, Pitch, Yaw)
    euler_angles_rad = relative_rotation.as_euler('zyx')

    # Extract roll, pitch, and yaw
    roll, pitch, yaw = euler_angles_rad

    # dsiatnce
    x = poses[i,0] - poses[j,0]
    y = poses[i,1] - poses[j,1]
    d = np.linalg.norm(poses[i,:3] - poses[j,:3])


    return x, y, yaw

import random
def modify(x, y):
    # Mean and standard deviation for the Gaussian distribution
    mean = 0.0  # Mean of the distribution
    stddev = 0.02  # Standard deviation of the distribution

    random_float1 = random.gauss(mean, stddev)
    random_float1 = max(-0.1, min(0.1, random_float1))

    random_float2 = random.gauss(mean, stddev)
    random_float2 = max(-0.1, min(0.1, random_float2))

    x += random_float1
    y += random_float2
    return x, y

# model stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt_path = '/media/aneesh/Ubuntu_storage/RRC/LIO-SAM-FinderNet-project/ansh_sync/weights/best.pt'

end2end = end2endModel()
end2end = nn.DataParallel(end2end)
end2end.load_state_dict(torch.load(ckpt_path))
print(" Model Loaded " + ckpt_path ,  flush=True)
end2end.to(device)

end2end.eval()

# get all poses and distances
f = open('/media/aneesh/Ubuntu_storage/RRC/LIO-SAM-FinderNet-project/ansh_sync/gt/dem_pose_all.csv', 'r')

files = []
poses = []

for line in f.readlines():
    k = line.split(',')
    files.append("../" + k[0])

    poses.append(
        np.array([k[i]
                for i in range(1,8)],
                dtype=np.float32)
    )
poses = np.stack(poses)
places = poses[:, :3]

all_dists = places.reshape(1, -1, 3) - places.reshape(-1, 1, 3)
all_dists = np.linalg.norm(all_dists, axis=-1)
print(all_dists.shape)

D_threshold = 170
cartesian_threshold = 0.35
loop_threshold = 0.08

with torch.no_grad():
    dists = []
    true_dists = []

    closureFile = open('./loopClosures.txt', 'w')

    for i in tqdm(range(0,len(files),10)):
        k1 = readImg([files[i]]).unsqueeze(1)
        
        cands = np.argwhere(all_dists[i] < cartesian_threshold).squeeze()

        pcd1 = o3d.geometry.PointCloud()
        pcd1name = "../pcd/" + files[i].split('/', 2)[-1].split('.')[0] + ".npy"
        pcd1points = np.load(pcd1name)
        pcd1.points = o3d.utility.Vector3dVector(pcd1points)

        for j in cands:
            if j == i:
                continue

            k2 = readImg([files[j]]).unsqueeze(1)
            dist, R, _, _ = end2end(k1, k2)

            pcd2 = o3d.geometry.PointCloud()
            pcd2name = "../pcd/" + files[i].split('/', 2)[-1].split('.')[0]  + ".npy"
            pcd2points = np.load(pcd2name)
            pcd2.points = o3d.utility.Vector3dVector(pcd2points)

            if dist.item() < 170:

                print("gentleman: ", files[i], files[j])
                icp_result = o3d.pipelines.registration.registration_icp(
                    pcd1, pcd2, max_correspondence_distance=0.05,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
                )

                tm = icp_result.transformation

                x = tm[0][-1]
                y = tm[1][-1]

                x, y, yaw = comparePose(poses, i, j)

                # print(files[i], " ", files[j])
                toWrite = (files[i].split('/')[2] + '/' + files[i].split('/')[3].split('_')[0]) \
                          + ',' \
                          + (files[j].split('/')[2] + '/' + files[j].split('/')[3].split('_')[0]) \
                          + ',' + str(x) + ',' + str(y) + ',' + str(R.item()) + '\n'
                closureFile.write(toWrite)

closureFile.close()                