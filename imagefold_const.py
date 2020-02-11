##only groups image with single classifications
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
datatab=pd.read_csv(inputdir+"Classifications.csv",delimiter=",")
classcol=np.array(datatab['CLASS'].tolist())
rowind=np.isin(classcol,np.array(['WRAP','WT','BULKY']))
datatabclean=datatab.loc[rowind]

# file_names=datatabclean.loc[:,'IMG']
classlab=np.unique(datatabclean['CLASS'].tolist())
# load resize image
for classele in classlab:
    os.makedirs(inputdir+'LApops_classify/'+classele,exist_ok=True)
    classcol=np.array(datatabclean['CLASS'].tolist())
    rowind=np.isin(classcol,classele)
    datatabsub=datatabclean.loc[rowind]
    files=datatabsub['IMG'].tolist()
    for file in files:
        file=file+'.tif'
        sourcfile=inputdir+"LApops_expand/"+file
        if os.path.isfile(sourcfile):
            shutil.copy(sourcfile,inputdir+'LApops_classify/'+classele+"/"+file)
