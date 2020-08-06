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

# For training data
inputdir="../data/"
testperc=[0.1,0.1]#valid and test
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
numsampvalid=math.floor(totsampsize*testperc[0])
numsamptest=math.floor(totsampsize*testperc[1])
sampleind=set(range(0,totsampsize))
testind=np.sort(np.array(random.sample(sampleind,numsamptest)))
validind=np.sort(np.array(random.sample(sampleind,numsampvalid)))
testindset=set(testind)
validindset=set(validind)
trainind=np.sort(np.array(list(sampleind.difference(testindset.union(validindset)))))

os.makedirs(inputdir+'LApops_classify',exist_ok=True)
os.makedirs(inputdir+'LApops_expand',exist_ok=True)
os.makedirs(inputdir+'LApops_classify/train',exist_ok=True)
os.makedirs(inputdir+'LApops_classify/test',exist_ok=True)
os.makedirs(inputdir+'LApops_classify/validate',exist_ok=True)
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
    os.makedirs(inputdir+'LApops_classify/validate/'+classele,exist_ok=True)
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
            
    for file in files[validind]:
        file=file+'.tif'
        sourcfile=inputdir+"LApops_expand/"+file
        if os.path.isfile(sourcfile):
            shutil.copy(sourcfile,inputdir+'LApops_classify/validate/'+classele+"/"+file)
        else:
            print('non existence file:'+file+'\n')

# For external test data set
inputdir="../data/LApops_new_Raw/"
outputdir="../data/LApops_new_test/test/"
classtypes=np.array(['WRAP','WT','BULKY'])
for classtype in classtypes:
    os.makedirs(outputdir+classtype,exist_ok=True)

infortab_list=[]
for batchdir in os.listdir(inputdir):
    if batchdir=='.DS_Store':
        continue
    
    datatab=pd.read_csv(inputdir+batchdir+"/Classifications.csv",delimiter=",")
    infortab_list.append(datatab)
    for rowi in range(datatab.shape[0]):
        sample=datatab.iloc[rowi,]
        shutil.copy(inputdir+batchdir+
        "/"+sample[0]+"/"+sample[1]+".tif",outputdir+sample[2]+"/"+sample[1]+".tif")
