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
import torch.nn as nn
from natsort import natsorted
from math import log2
import sys
from tqdm import tqdm
from scipy import ndimage

from myModels import *

from generateDEM import displayDEM

from PIL import Image

parser = argparse.ArgumentParser(description='Train Code for Spatial Transformer')
parser.add_argument('--data_path', help='Path to the dataset', default='/home/aneesh/UbuntuStorage/RRC/LIO-SAM-FinderNet-project/proc/matches/consolidated.txt')
parser.add_argument('--base_path', help='path to parent directory of the image dataset folder' , default='/home/aneesh/UbuntuStorage/RRC/LIO-SAM-FinderNet-project/proc')
parser.add_argument('--batch_size' , help='Size of Batch' , default= 4)
parser.add_argument('--num_epochs' , help='Number of epochs' , default = 200 )
parser.add_argument('--image_resolution' , help='Size of image ' , default =250)
parser.add_argument('--save_path', help='base path to save the model' , default='/home/aneesh/UbuntuStorage/RRC/LIO-SAM-FinderNet-project/proc/findernet/models')
parser.add_argument('--iters_per_ckpt' , help= 'number of iterations to save a checkpoint' , default=100)
parser.add_argument('--total_train_samples' , help= 'Total number of train samples' , default= 100 )
parser.add_argument('--total_test_samples' , help='Total number of validation/test samples' , default= 50)
parser.add_argument('--start_index' , help='enter row number of the csv to consider as start ', default=0)
parser.add_argument('--margin' , help ='margin of the triplet loss ', default= 2.75 )
parser.add_argument('--continue_train' , help =' Continue traiing from a previous checkpoint  ', default= False )
parser.add_argument('--path_to_prev_ckpt' , help =' path to the previous checkpoint only required if continue_train is true  ', default= 'UnrealModel/model_unreal_8.pt' )
parser.add_argument('--lr_change_frequency', help='Number of epochs to update the learning rate', default=10)
args = parser.parse_args()


''' return a minibatch containing each image stores at the paths in `paths` '''
def readImg(paths):
    imgs = []
    for i in range(len(paths)):
        # img = np.asarray(Image.open(paths[i])).astype(np.float32)
        img = np.load(paths[i]).astype(np.float32)
        img = cv2.resize(img, (500,500), interpolation=cv2.INTER_NEAREST)

        # imgs.append(img.astype(np.uint8))
        imgs.append(img)
    
    imgs = torch.tensor(np.array(imgs), dtype=torch.float).to(device)
    return imgs

''' torch init '''
torch.autograd.set_detect_anomaly(True)
SEED = 123456789
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True ## makes the process deterministic i.e when a same input is given to a the same algortihm on the same hardware the result is the same
torch.backends.cudnn.benchmark = False ## It enables auto tune to find the best algorithm for the given hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Total samples: " + str(args.total_train_samples) , flush=True)
numBatches = math.ceil(args.total_train_samples/args.batch_size)
numValBatches = math.ceil(args.total_test_samples/args.batch_size)

def CreateBatchData(file, start_index=0, mode='train'):
	if(mode == 'train'):
		final = args.total_train_samples
	if(mode == 'validation'):
		final = args.total_test_samples

	EndIndex = start_index + final #args.total_train_samples

	# AnchorSamples = DF['anchor'][start_index:EndIndex]
	# PositiveSamples = DF['positive'][start_index:EndIndex]
	# NegativeSamples = DF['negative'][start_index:EndIndex]

	f = open(file, 'r')
	demLists = []
	for l in f.readlines():
		demLists.append(l.split(' '))
		demLists[-1][2] = demLists[-1][2][:-1]
	demLists = np.array(demLists)
	f.close()

	AnchorSamples = list(demLists[start_index : start_index + final, 0])
	PositiveSamples = list(demLists[start_index : start_index + final, 1])
	NegativeSamples = list(demLists[start_index : start_index + final, 2])

	AnchorDataSet = []
	PositiveDataSet = []
	NegativeDataSet  = []
    
	for i in range(start_index, start_index + final, args.batch_size):
		# print("From ", i, " to ", min(len(PositiveSamples), i + args.batch_size))

		a = AnchorSamples[i : min(len(AnchorSamples), i + args.batch_size)]
		p = PositiveSamples[i : min(len(PositiveSamples), i + args.batch_size)]
		n = NegativeSamples[i : min(len(NegativeSamples), i + args.batch_size)]

		for k, (x,y,z) in enumerate(zip(a,p,n)):
			a[k] = os.path.join(args.base_path, x)
			p[k] = os.path.join(args.base_path, y)
			n[k] = os.path.join(args.base_path, z)

		AnchorDataSet.append(a)
		PositiveDataSet.append(p)
		NegativeDataSet.append(n)

	return AnchorDataSet , PositiveDataSet , NegativeDataSet

end2end = end2endModel()

# n = readImg(anchorDS[10])
# print(anchorDS[10])

# a = np.load(anchorDS[0][0])
# a = cv2.resize(a, (500,500), interpolation=cv2.INTER_NEAREST)
# plt.imshow(a)
# plt.show()


# plt.imshow(n[0].detach().cpu())
# plt.show()

# displayDEM(np.array(n[0].detach().cpu()))


mseLoss = nn.MSELoss()

if(args.continue_train):
	end2end.load_state_dict(torch.load(args.path_to_prev_ckpt) )
	print(" Model Loaded " + str(args.path_to_prev_ckpt) ,  flush=True)
	
'''
Init loading, scheduler optimizer etc.
'''
print(torch.cuda.device_count())
end2end.to(device)

optimizer = optim.Adam(end2end.parameters())
save_checkpoint = 0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_change_frequency, gamma=0.1)

Zeros = torch.zeros((args.batch_size ,1)).to("cuda")

saveCounter = 0
for e in range(args.num_epochs):
    epochLoss = 0
    correct = 0
    validatonCorrect = 0
    cumulativeLoss1 = 0
    cumulativeLoss2 = 0

    '''read the list of anchor-positive-negative pairs created by `createKittiTrainList.py`'''
    # DF = pd.read_csv( args.data_path  )
    # DF = DF.sample(frac=1).reset_index(drop=True)

    '''create datasets containing minibatches of size 12'''
    # AnchorDataSet , PositiveDataSet , NegativeDataSet = CreateBatchData(args.start_index, 'train' , DF)
    AnchorDataSet , PositiveDataSet , NegativeDataSet = CreateBatchData('matches/consolidated.txt')

    '''iterate over all batches'''
    for batch_num in tqdm(range(numBatches)):
        optimizer.zero_grad()
        if(len( AnchorDataSet[batch_num]) > 0):
            AnchorImgs = readImg(AnchorDataSet[batch_num])
            PositiveImgs = readImg(PositiveDataSet[batch_num])
            NegativeImgs = readImg(NegativeDataSet[batch_num])
	    
            for i in range(args.batch_size):
                print("triplet ", i)
                displayDEM(AnchorImgs[i].detach().cpu())
                displayDEM(PositiveImgs[i].detach().cpu())
                displayDEM(NegativeImgs[i].detach().cpu())
	    
            '''images need a channel, make 12x250x250 -> 12x1x250x250s'''
            AnchorImgs = AnchorImgs.unsqueeze(1)
            PositiveImgs = PositiveImgs.unsqueeze(1)
            NegativeImgs = NegativeImgs.unsqueeze(1)
	    
            '''Recover anchor, positive and negative embs. Also get reconstructed pcds, to compuite enc-dec loss'''
            '''Compute distance between anchor-positive / anchor-negative embs, feed this to loss laster'''
            Scores_ap,  _, Reconstructed_DEM_a1 , Reconstructed_DEM_p  = end2end.forward(AnchorImgs , PositiveImgs)
            Scores_an,  _, Reconstructed_DEM_a2 , Reconstructed_DEM_n  = end2end.forward(AnchorImgs , NegativeImgs)
            
            '''triplet loss of distance scores'''
            # print("apshape!: ", Scores_ap.shape)
            # print("anshape!: ", Scores_ap.shape)
	    
            # print(",argin:." , args.margin)
            # print(",argin:." , Zeros)
	    
            Zeros = torch.zeros((Scores_ap.shape[0], 1)).to("cuda")
            Loss1 = torch.sum( torch.maximum( Scores_ap -  Scores_an + args.margin , Zeros).to(device))
            
            '''dem enc-dec loss'''
            Loss2 = (mseLoss(Reconstructed_DEM_a1 , AnchorImgs ) +  mseLoss(Reconstructed_DEM_a2 , AnchorImgs) 
                    + mseLoss(Reconstructed_DEM_p , PositiveImgs ) + mseLoss(Reconstructed_DEM_n , NegativeImgs))
	    
            cumulativeLoss1 += Loss1
            cumulativeLoss2 += 0.1*Loss2
	    
            '''backprop'''
            # Loss = Loss1 + 0.1*Loss2  
            # print(Loss2)
            Loss = Loss2
            
            Loss.backward()
            # print("grad: ", torch.autograd.grad(Loss2, Reconstructed_DEM_a1))
	    
            optimizer.step()
            epochLoss += Loss
	    
            # exit(0)
	    
    saveCounter += 1
    print( "Epoch Loss  =  "  + str(epochLoss)  +  " classification = " + str(cumulativeLoss1) + " reconstruction = " + str(cumulativeLoss2) + " Epoch Number =  " + str(e) ,  flush=True )
    if(saveCounter % 3 == 0):
        save_file_name = "model_" + "epoch_num_" + str(e) + "_Loss_" + str(epochLoss) +  ".pt"
        save_file_name = os.path.join( args.save_path , save_file_name)
        torch.save(end2end.state_dict(), save_file_name)