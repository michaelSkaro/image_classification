##this script set up the original trained model
##load the model
##plot corresponding figures
import argparse
import os
import random
import shutil
import time
import warnings
import pickle
# import feather
import numpy as np
import math
import sys
import copy
import h5py
import re
import pandas as pd
import matplotlib.pyplot as plt

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
from image_classify_transfer import train, test, parse_func_wrap,visualize_model

# sys.path.insert(1,'/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/nc_model/nnt/model_training/')
# import nnt_struc as models
listmodelfile={
    "resnet18": ('resnet18-5c106cde.pth',str),
    "resnet34": ('resnet34-333f7ec4.pth',str),
    "resnet50": ('resnet50-19c8e357.pth',str),
    "resnet101": ('resnet101-5d3b4d8f.pth',str),
    "resnet152": ('resnet152-b121ed2d.pth',str)
}
data_transforms = {
    'train': transforms.Compose([
        transforms.Pad((0,174,0,174),fill=(255,255,255)),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=360),
        # transforms.RandomGrayscale(),
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

def test(args,model,test_loader,device):
    model.eval()
    test_loss=0.0
    running_corrects=0
    with torch.no_grad():
        for data, target in test_loader:
            # if args.gpu is not None:
            #     data=data.cuda(args.gpu,non_blocking=True)
            # target=target.cuda(args.gpu,non_blocking=True)
            data,target=data.to(device),target.to(device)
            output=model(data)
            _,preds=torch.max(output,1)
            # loss=F.nll_loss(output,target)
            loss=F.cross_entropy(output,target,reduction='mean')
            test_loss+=loss.item()*data.size(0)
            running_corrects+=torch.sum(preds==target.data)
    
    epoch_loss=test_loss/dataset_sizes['test']
    epoch_acc=running_corrects.double()/dataset_sizes['test']
    print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss,epoch_acc))
    return epoch_loss

def visualize_model(model,dataloaderslc,num_images=6):
    #from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    was_training=model.training
    model.eval()
    images_so_far=0
    fig=plt.figure()

    with torch.no_grad():
        for i, (inputs,labels) in enumerate(dataloaderslc):
            inputs=inputs.to(device)
            labels=labels.to(device)

            outputs=model(inputs)
            _, preds=torch.max(outputs,1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax=plt.subplot(num_images//2,2,images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}, labels: {}'.format(class_names[preds[j]],class_names[labels[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far==num_images:
                    model.train(mode=was_training)
                    return

# import the model from model script
random.seed(1)
torch.manual_seed(1)
inputdir="./"
os.chdir(inputdir)
##load information table
infortab=pd.read_csv(inputdir+'submitlist.tab',sep="\t",header=0)
infortab=infortab.astype({"batch_size": int,"test_batch_size": int})

global dataset_sizes
##check validate data set
rowi=7
device=torch.device('cpu')
##Loading data
loaddic=torch.load("./res/"+str(rowi)+"/model_best.resnetode.tar",map_location=device)
args=loaddic["args_input"]
image_datasets={x: datasets.ImageFolder(os.path.join("../../data/","LApops_classify",x),data_transforms[x]) for x in ['train','validate','test']}
dataloaders={x: torch.utils.data.DataLoader(image_datasets[x],batch_size=args.batch_size,shuffle=True, num_workers=0) for x in ['train','validate','test']}
dataset_sizes={x: len(image_datasets[x]) for x in ['train','validate','test']}
class_names=image_datasets['train'].classes
model_ft=models.__dict__[args.net_struct]()
num_ftrs=model_ft.fc.in_features
model_ft.fc=nn.Linear(num_ftrs,3)
print(args.net_struct)
model_ft=torch.nn.DataParallel(model_ft)
if args.pretrained==1:
    model_ft.load_state_dict(loaddic['state_dict'])
acc1=test(args,model_ft,dataloaders['test'],device)
acc1=test(args,model_ft,dataloaders['validate'],device)

visualize_model(model_ft,dataloaders['test'])
visualize_model(model_ft,dataloaders['validate'])
