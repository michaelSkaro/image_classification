##just run the following Line
##transfer learning from pretrained resnet18 on imagenet.
## it runs with two folder on dropbox needed emi_nnt_image/data/LApops_classify and emi_nnt_image/pretrained
## to test it with gpu --gpu-use 1
## it runs on cpu
time python3 image_classify_transfer.py  --batch-size 4 --test-batch-size 4 --epochs 25 --learning-rate 0.001 --seed 1 --gpu-use 0 > testmodel.1.out
