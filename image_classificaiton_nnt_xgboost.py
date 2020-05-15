##this training script support
## this script use resnet as a feature extration tool and do the classification on xgboost based on extracted features
## tutorial on transfer learning https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
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
from nltk import flatten
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
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.resnet import model_urls

import xgboost as xgb #xgboost
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder #encoding string
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, RandomizedSearchCV # cv and parameter search
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# model_urls['resnet18']=model_urls['resnet18'].replace('https://', 'http://')

listmodelfile={
    "resnet18": ('resnet18-5c106cde.pth',str),
    "resnet34": ('resnet34-333f7ec4.pth',str),
    "resnet50": ('resnet50-19c8e357.pth',str),
    "resnet101": ('resnet101-5d3b4d8f.pth',str),
    "resnet152": ('resnet152-b121ed2d.pth',str)
}
# samplewholeselec=list(range(9995,10000))## the whole time series just for testing
##default parameters
args_internal_dict={
    "batch_size": (4,int),
    "test_ratio": (0.2,float), # ratio of sample for test in each epoch
    "test_batch_size": (4,int),
    "epochs": (20,int),
    "learning_rate": (0.001,float),
    "momentum": (0.9,float),
    "no_cuda": (False,bool),
    "seed": (1,int),
    "log_interval": (10,int),
    "net_struct": ("resnet18",str),
    "optimizer": ("adam",str),##adam
     "p": (0.0,float),
     "gpu_use": (1,int),# whehter use gpu 1 use 0 not use
     "freeze_till": ('layer4',str), #till n block ResNet18 has 10 blocks
     "pretrained": (1,int)
     # "groupsingle": (1,int)# 1 only classify figures with single groups. 0 Not supported yet(classify figures with multiple groups)
}
###fixed parameters: for communication related parameter within one node
fix_para_dict={#"world_size": (1,int),
               # "rank": (0,int),
               # "dist_url": ("env://",str),#"tcp://127.0.0.1:FREEPORT"
               "gpu": (None,int),
               # "multiprocessing_distributed": (False,bool),
               # "dist_backend": ("nccl",str), ##the preferred way approach of parallel gpu
               "workers": (1,int)
}
##image transformation
data_transforms = {
    'train': transforms.Compose([
        transforms.Pad((0,174,0,174),fill=(255,255,255)),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=360,fill=(255,255,255)),
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
inputdir="../data/"
def train(args,model,train_loader,optimizer,epoch,device):
    model.train()
    trainloss=0.0
    running_corrects=0
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("checkerstart")
        # if args.gpu is not None:
        #     data=data.cuda(args.gpu,non_blocking=True)
        #
        # target=target.cuda(args.gpu,non_blocking=True)
        data,target=data.to(device),target.to(device)
        output=model(data)
        _,preds=torch.max(output,1)
        # loss=F.nll_loss(output,target)
        loss=F.cross_entropy(output,target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()*data.size(0)
        corrclas=torch.sum(preds==target.data)
        running_corrects+=corrclas
        if batch_idx % args.log_interval==0:
            lrstr=' lr: '+str(get_lr(optimizer))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss(per sample): {:.6f} accuracy {:.2f} lr{}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100. * batch_idx*len(data)/len(train_loader.dataset),loss.item(),corrclas.double()/args.batch_size,lrstr))
    
    epoch_loss=trainloss/dataset_sizes['train']
    epoch_acc=running_corrects.double()/dataset_sizes['train']
    
    print('END Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss,epoch_acc))
    
    return epoch_loss

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

def parse_func_wrap(parser,termname,args_internal_dict):
    commandstring='--'+termname.replace("_","-")
    defaulval=args_internal_dict[termname][0]
    typedef=args_internal_dict[termname][1]
    parser.add_argument(commandstring,type=typedef,default=defaulval,
                        help='input '+str(termname)+' for training (default: '+str(defaulval)+')')
    
    return(parser)

def save_checkpoint(state,is_best,is_best_train,filename='checkpoint.resnetode.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'model_best.resnetode.tar')
    
    if is_best_train:
        shutil.copyfile(filename,'model_best_train.resnetode.tar')

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

def visualize_model(model,dataloaders,num_images=6):
    #from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    was_training=model.training
    model.eval()
    images_so_far=0
    fig=plt.figure()

    with torch.no_grad():
        for i, (inputs,labels) in enumerate(dataloaders):
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

def visualize_model_nonnnt(model,ntextractor,dataloadersloc,num_images=6):
    #visulization function for sktlearn models
    was_training=ntextractor.training
    ntextractor.eval()
    images_so_far=0
    fig=plt.figure()
    for i, (inputs,labels) in enumerate(dataloadersloc):
        inputs=inputs.to(device)
        labels=labels.to(device)
        extrfeature=ntextractor(inputs)
        extrfeature=extrfeature.detach()
        extrfeature=torch.flatten(extrfeature,start_dim=1)
        extrfeature=extrfeature.to('cpu')
        preds=model.predict(extrfeature)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax=plt.subplot(num_images//2,2,images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}, labels: {}'.format(class_names[preds[j]],class_names[labels[j]]))
            imshow(inputs.cpu().data[j])
            if images_so_far==num_images:
                plt.savefig("exam_fig"+str(num_images)+".pdf")
                ntextractor.train(mode=was_training)
                return

def get_lr(optimizer):#output the lr as scheduler is used
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    # Training settings load-in through command line
    parser=argparse.ArgumentParser(description='PyTorch Example')
    for key in args_internal_dict.keys():
        parser=parse_func_wrap(parser,key,args_internal_dict)
    
    for key in fix_para_dict.keys():
        parser=parse_func_wrap(parser,key,fix_para_dict)
    
    args=parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic=True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    # args.distributed=args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node=torch.cuda.device_count()
    # ngpus_per_node=1
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    ## args.world_size=ngpus_per_node*args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    ## arg_transf=copy.deepcopy(args)
    ## arg_transf.ngpus_per_node=ngpus_per_node
    ## mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
    main_worker(args.gpu,ngpus_per_node,args)

def main_worker(gpu,ngpus_per_node,args):
    global best_acc1
    global dataset_sizes
    # args.gpu=gpu
    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))
    
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    ## args.rank=args.rank*ngpus_per_node+gpu
    ## dist.init_process_group(backend=args.dist_backend,init_method="env://",#args.dist_url,
    ## world_size=args.world_size,rank=args.rank)
    
    image_datasets={x: datasets.ImageFolder(os.path.join(inputdir,"LApops_classify",x),data_transforms[x]) for x in ['train','validate','test']}
    dataloaders={x: torch.utils.data.DataLoader(image_datasets[x],batch_size=args.batch_size,shuffle=True, num_workers=args.workers) for x in ['train','validate','test']}
    dataset_sizes={x: len(image_datasets[x]) for x in ['train','validate','test']}
    class_names=image_datasets['train'].classes
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ##quick view the image
    # inputs,classes=next(iter(dataloaders['train']))
    # out=torchvision.utils.make_grid(inputs)
    # imshow(out,title=[class_names[x] for x in classes])
    
    ##store data
    with open("pickle_dataloaders.dat","wb") as f1:
        pickle.dump(dataloaders,f1,protocol=4)
    
    dimdict={
        "dataset_sizes": (dataset_sizes,int),
    }
    with open("pickle_dimdata.dat","wb") as f3:
        pickle.dump(dimdict,f3,protocol=4)
        
    # model.eval()
    # if args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model.cuda(args.gpu)
    #     # When using a single GPU per process and per
    #     # DistributedDataParallel, we need to divide the batch size
    #     # ourselves based on the total number of GPUs we have
    #     args.batch_size=int(args.batch_size/ngpus_per_node)
    #     args.workers=int((args.workers+ngpus_per_node-1)/ngpus_per_node)
    #     model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # else:
    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    
    # model_ft=models.resnet18()
    model_ft=models.__dict__[args.net_struct]()
    print(args.net_struct)
    if args.pretrained==1:
        model_ft.load_state_dict(torch.load('../pretrained/'+listmodelfile[args.net_struct][0]))
        
    num_ftrs=model_ft.fc.in_features
    model_ft.fc=nn.Linear(num_ftrs,3)
    if args.pretrained==1:
        paracounter=0
        for name, param in model_ft.named_parameters():
            if not bool(re.search(args.freeze_till,name)):
                # print(name)
                param.requires_grad=False
            else:
                break
            paracounter=paracounter+1
    
    # counti=0
    # for name, param in model_ft.named_parameters():#len(model_ft.state_dict())
    #         print(name)
    #         counti=counti+1
    
    # match='layer4.0.conv1.weight'
    # counti=1
    # for name, param in model_ft.named_parameters():
    #     # print(name)
    #     if name==match:
    #         print(counti)
    #     counti=counti+1
    
    model_ft=torch.nn.DataParallel(model_ft)
    if args.gpu_use==1:
        device=torch.device("cuda:0")#cpu
    else:
        device=torch.device("cpu")
    
    model_ft.to(device)
    if args.optimizer=="sgd":
        optimizer=optim.SGD(model_ft.parameters(),lr=args.learning_rate,momentum=args.momentum)
    elif args.optimizer=="adam":
        optimizer=optim.Adam(model_ft.parameters(),lr=args.learning_rate)
    
    # optimizer=optim.Adam(model_ft.parameters(),lr=args.learning_rate)
    ## lr decay scheduler
    # scheduler=lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5)
    cudnn.benchmark=True
    ##model training
    for epoch in range(1,args.epochs+1):
        acctr=train(args,model_ft,dataloaders['train'],optimizer,epoch,device)
        acc1=test(args,model_ft,dataloaders['test'],device)
        if scheduler is not None:
            scheduler.step(acc1)
        # test(args,model,traindataloader,device,ntime) # to record the performance on training sample with model.eval()
        if epoch==1:
            best_acc1=acc1
            best_train_acc=acctr
        
        # is_best=acc1>best_acc1
        is_best=acc1<best_acc1
        is_best_train=acctr<best_train_acc
        # best_acc1=max(acc1,best_acc1)
        best_acc1=min(acc1,best_acc1)
        best_train_acc=min(acctr,best_train_acc)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_struct,
            'state_dict': model_ft.state_dict(),
            'best_acc1': best_acc1,
            'best_acctr': best_train_acc,
            'optimizer': optimizer.state_dict(),
            'args_input': args,
        },is_best,is_best_train)
        # device=torch.device('cpu')
        # # model=TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load('./1/checkpoint.resnetode.tar',map_location=device))
    
    # visualize_model(model_ft,dataloaders['test'])
    acc1=test(args,model_ft,dataloaders['validate'],device)
    
    ##add xgboost here
    model_ft2=copy.deepcopy(model_ft)
    model_ft2.module=nn.Sequential(*list(model_ft2.module.children())[:-1])#remove last fc layer
    model_ft2.eval()
    traintarget=torch.tensor([],dtype=torch.long,device=device)
    trainfeature=torch.tensor([],device=device)
    testtarget=torch.tensor([],dtype=torch.long,device=device)
    testfeature=torch.tensor([],device=device)
    validatetarget=torch.tensor([],dtype=torch.long,device=device)
    validatefeature=torch.tensor([],device=device)
    with torch.no_grad():
        for data, target in dataloaders['train']:
            data,target=data.to(device),target.to(device)
            feature=model_ft2(data)
            traintarget=torch.cat([traintarget,target],0)
            trainfeature=torch.cat([trainfeature,torch.flatten(feature,start_dim=1)],0)
        for data, target in dataloaders['test']:
            data,target=data.to(device),target.to(device)
            feature=model_ft2(data)
            testtarget=torch.cat([testtarget,target],0)
            testfeature=torch.cat([testfeature,torch.flatten(feature,start_dim=1)],0)
        for data, target in dataloaders['validate']:
            data,target=data.to(device),target.to(device)
            feature=model_ft2(data)
            validatetarget=torch.cat([validatetarget,target],0)
            validatefeature=torch.cat([validatefeature,torch.flatten(feature,start_dim=1)],0)
    
    traintarget,trainfeature,testtarget,testfeature,validatetarget,validatefeature=traintarget.to('cpu'),trainfeature.to('cpu'),testtarget.to('cpu'),testfeature.to('cpu'),validatetarget.to('cpu'),validatefeature.to('cpu')
    traintest_x=torch.cat([trainfeature,testfeature],0)
    traintest_y=torch.cat([traintarget,testtarget],0)
    
    ##hyperparameter searching
    # traintest_y_onehot=preprocessing.label_binarize(traintest_y,classes=[0,1,2])
    params={
        'n_estimators': [100,300,1000],
        'min_child_weight': [1,5,10],
        'subsample': [0.6,0.8,1.0],
        'colsample_bytree': [0.6,0.8,1.0],
        'max_depth': [3,6,9],
        'learning_rate': [0.01,0.05,0.1,0.3],
        'gamma': [0,0.5,1]
        }
    gbm=xgb.XGBClassifier(max_depth=6,n_estimators=100,num_class=3,objective='multi:softmax')
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=args.seed)
    #random search
    random_search=RandomizedSearchCV(gbm,param_distributions=params,n_iter=10, n_jobs=3,cv=skf.split(traintest_x,traintest_y),verbose=3,random_state=args.seed,scoring='accuracy')#scoring='roc_auc_ovr'
    random_search.fit(traintest_x,traintest_y)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print(random_search.best_score_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    results=pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv',index=False)
    
    ##brutle search takes too long
    # grid=GridSearchCV(estimator=gbm,param_grid=params,scoring='accuracy',n_jobs=3, cv=skf.split(traintest_x,traintest_y),verbose=3)
    # grid.fit(traintest_x,traintest_y)
    # print('\n Best estimator:')
    # print(grid.best_estimator_)
    # print('\n Best score:')
    # print(grid.best_score_)
    # print('\n Best parameters:')
    # print(grid.best_params_)
    # results = pd.DataFrame(grid.cv_results_)
    # results.to_csv('xgb-grid-search-results-01.csv',index=False)
    # y_test = grid.best_estimator_.predict_proba(test)
    # results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})
    # results_df.to_csv('submission-grid-search-xgb-porto-01.csv', index=False)
    
    ##train the model by the chosen parameter sets
    #### crosss validation training on xgboost
    # 'n_estimators': [100,300,1000],
    # 'min_child_weight': [1,5,10],
    # 'subsample': [0.6,0.8,1.0],
    # 'colsample_bytree': [0.6,0.8,1.0],
    # 'max_depth': [3,6,9],
    # 'learning_rate': [0.01,0.05,0.1,0.3],
    # 'gamma': [0,0.5,1]
    
    kf=KFold(n_splits=10,shuffle=True,random_state=args.seed)
    for train_index,test_index in kf.split(traintest_x):
        gbm=xgb.XGBClassifier(max_depth=9,n_estimators=100,num_class=3,objective='multi:softmax',subsample=0.6,min_child_weight=5,learning_rate=0.01,gamma=0,colsample_bytree=0.6).fit(traintest_x[train_index,],traintest_y[train_index])
        predictions=gbm.predict(traintest_x[test_index,])
        actuals=traintest_y[test_index]
        print(accuracy_score(actuals,predictions))
    
    gbm=xgb.XGBClassifier(max_depth=9,n_estimators=100,num_class=3,objective='multi:softmax',subsample=0.6,min_child_weight=5,learning_rate=0.05,gamma=0,colsample_bytree=0.6).fit(traintest_x,traintest_y)
    predictions=gbm.predict(validatefeature)
    print(accuracy_score(validatetarget,predictions))
    gbm.save_model('xgboostmodel.json')
    ##roc curve plot
    nclass=3
    traintest_y_bin=label_binarize(traintest_y,classes=[0,1,2])
    validate_y_bin=label_binarize(validatetarget,classes=[0,1,2])
    gbm_one=OneVsRestClassifier(xgb.XGBClassifier(max_depth=9,n_estimators=100,num_class=3,objective='multi:softmax',subsample=0.6,min_child_weight=5,learning_rate=0.05,gamma=0,colsample_bytree=0.6))
    validate_y_predict=gbm_one.fit(traintest_x,traintest_y).predict_proba(validatefeature)
    validate_y_predict[np.isnan(validate_y_predict)]=0
    # Compute ROC curve and ROC area for each class
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(nclass):
        fpr[i], tpr[i], _ = roc_curve(validate_y_bin[:,i],validate_y_predict[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(validate_y_bin.ravel(), validate_y_predict.ravel())
    roc_auc["micro"]=auc(fpr["micro"],tpr["micro"])
    plt.figure()
    classi=1#0 1 2
    lw=2
    plt.plot(fpr[classi],tpr[classi],color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' % roc_auc[classi])
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("ROC_class"+str(classi)+".pdf")
    ##show a few examples
    visualize_model_nonnnt(gbm,model_ft2,dataloaders['train'],6)
    visualize_model_nonnnt(gbm,model_ft2,dataloaders['test'],6)
if __name__ == '__main__':
    main()
