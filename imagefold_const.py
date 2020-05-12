##only groups image with single classifications
##the training and testing sample will be constructed so that each class is with equal number of samples in training and validation group
## selecting into training+validation and validation group are random
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
from nltk import flatten
import re
import pandas as pd
from collections import Counter

import torch
import torch.nn.parallel
import torch.utils.data as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import datasets, models, transforms
inputdir="../data/"
testperc=0.2
datatab=pd.read_csv(inputdir+"Classifications.csv",delimiter=",")
classcol=np.array(datatab['CLASS'].tolist())
rowind=np.isin(classcol,np.array(['WRAP','WT','BULKY']))
datatabclean=datatab.loc[rowind]
nonduplicated_ind=[not ele for ele in datatabclean['IMG'].duplicated(keep=False).tolist()]
datatabclean2=datatabclean.loc[nonduplicated_ind]

# file_names=datatabclean.loc[:,'IMG']
classlab=np.unique(datatabclean2['CLASS'].tolist())
counterlist=Counter(datatabclean2['CLASS'].tolist())
totsampsize=min(counterlist.values())# for each class
numsamptest=math.floor(totsampsize*testperc)
sampleind=set(range(0,totsampsize))
testind=np.sort(np.array(random.sample(sampleind,numsamptest)))
trainind=np.sort(np.array(list(sampleind.difference(set(testind)))))

os.makedirs(inputdir+'LApops_classify',exist_ok=True)
os.makedirs(inputdir+'LApops_expand',exist_ok=True)
os.makedirs(inputdir+'LApops_classify/train',exist_ok=True)
os.makedirs(inputdir+'LApops_classify/test',exist_ok=True)
for direle in os.listdir(inputdir+'LApops_Raw'):
    if os.path.isdir(inputdir+'LApops_Raw/'+direle):
        for fileele in os.listdir(inputdir+'LApops_Raw/'+direle):
            sourcfile=inputdir+'LApops_Raw/'+direle+'/'+fileele
            targetfile=inputdir+'LApops_expand/'+fileele
            shutil.copy(sourcfile,targetfile)

random.seed(1)
for classele in classlab:
    os.makedirs(inputdir+'LApops_classify/train/'+classele,exist_ok=True)
    os.makedirs(inputdir+'LApops_classify/test/'+classele,exist_ok=True)
    classcol=np.array(datatabclean2['CLASS'].tolist())
    rowind=np.isin(classcol,classele)
    datatabsub=datatabclean2.loc[rowind]
    totsamp_ind=np.array(random.sample(list(range(0,datatabsub.shape[0])),totsampsize))
    datatabsub2=datatabsub.iloc[totsamp_ind]
    files=np.array(datatabsub2['IMG'].tolist())
    for file in files[trainind]:
        file=file+'.tif'
        sourcfile=inputdir+"LApops_expand/"+file
        if os.path.isfile(sourcfile):
            shutil.copy(sourcfile,inputdir+'LApops_classify/train/'+classele+"/"+file)
        else:
            print('non existence file:'+file+'\n')
    
    for file in files[testind]:
        file=file+'.tif'
        sourcfile=inputdir+"LApops_expand/"+file
        if os.path.isfile(sourcfile):
            shutil.copy(sourcfile,inputdir+'LApops_classify/test/'+classele+"/"+file)
        else:
            print('non existence file:'+file+'\n')
