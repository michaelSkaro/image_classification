# *Neurospora crassa* microscopic image classification by transfer learning

This repository keep the script and function used to classify *Neurospora crassa* images into WT, BULKY and WARP. The image_classify_transfer.py file is used for model training and res_visual.py is used for result visualization. To train the model as done in the paper, please do:

```
time python3 image_classify_transfer.py  --batch-size 15 --test-batch-size 15 --epochs 500 --learning-rate 0.001 --seed 2 --net-struct resnet50 --optimizer sgd   --pretrained 1 --freeze-till layer3  1>> ./model.out 2>> ./model.err
```

Before you run this file, please make sure you have gpu to run, have the image data for training, and have the pretrained model ready.
