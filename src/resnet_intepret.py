## this script set up the original trained model
## load the model
## do model feature intepretation based on NoiseTunnel + IntegratedGradients
## result is absolute value to show the important part
### this is run on local mac computer with pytorch 1.2.0, torchvision 0.4.0a0+6b959ee, and cpu
import argparse
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
import math
import sys
import copy
import h5py
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import Saliency
data_transforms = {
    'train': transforms.Compose([
        transforms.Pad((0,174,0,174),fill=(255,255,255)),
        transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=360),#,fill=(255,255,255) this is disabled because of the version of torchvision
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Pad((0,174,0,174),fill=(255,255,255)),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'validate': transforms.Compose([
        transforms.Pad((0,174,0,174),fill=(255,255,255)),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}
def imshow(inp, title=None):
    """Imshow for Tensor."""
    #from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    inp=inp.numpy().transpose((1,2,0))
    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])
    inp=std*inp+mean
    inp=np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# import the model from model script
seed=2
random.seed(seed)
torch.manual_seed(seed)
inputdir="LOCAL_PROJ_FOLDER"
os.chdir(inputdir)

global dataset_sizes
##check validate data set
rowi=5#model 5 is the final chosen model
device=torch.device('cpu')
##Loading data
loaddic=torch.load("./res/"+str(rowi)+"/model_best.resnetode.tar",map_location=device)
args=loaddic["args_input"]
image_datasets={x: datasets.ImageFolder(os.path.join(inputdir,'data/LApops_classify_intepret/',x),data_transforms[x]) for x in ['test']}
dataloaders={x: torch.utils.data.DataLoader(image_datasets[x],batch_size=1,shuffle=True, num_workers=0) for x in ['test']}#get one figure one time
dataset_sizes={x: len(image_datasets[x]) for x in ['test']}
class_names=image_datasets['test'].classes
model_ft=models.__dict__[args.net_struct]()
num_ftrs=model_ft.fc.in_features
model_ft.fc=nn.Linear(num_ftrs,3)
print(args.net_struct)
model_ft=torch.nn.DataParallel(model_ft)
model_ft.load_state_dict(loaddic['state_dict'])
model_ft.eval()
default_cmap=LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
baseline=2.6400# ~maximum in the image, white place
for batch_idx, (input, target) in enumerate(dataloaders['test']):
     image_input=input.cpu().data[0].numpy().transpose((1,2,0))
     mean=np.array([0.485,0.456,0.406])
     std=np.array([0.229,0.224,0.225])
     image_input=std*image_input+mean
     image_input=np.clip(image_input,0,1)
     output=model_ft(input)
     prediction_score,pred_label_idx=torch.topk(output,1)
     pred_clname=class_names[pred_label_idx]
     integrated_gradients=IntegratedGradients(model_ft)
     noise_tunnel=NoiseTunnel(integrated_gradients)
     attributions_ig_nt=noise_tunnel.attribute(input,n_samples=10,nt_type='smoothgrad_sq', target=pred_label_idx,baselines=baseline)
     pdf=PdfPages(inputdir+"res/image_intepret_"+str(batch_idx)+".pdf")
     fig, axis= viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                           image_input,
                                           ["original_image", "heat_map"],
                                           ["all", "positive"],
                                           cmap=default_cmap,
                                           show_colorbar=True,
                                           titles=['Truth: '+class_names[target]+' Prediction: '+pred_clname,'NoiseTunnel'])
     pdf.savefig(fig)
     plt.close('all')
     pdf.close()
