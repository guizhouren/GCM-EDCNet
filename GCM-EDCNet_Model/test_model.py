from DenseNet_xgroup_senet import dense_xgroup_senet
import torch
from train__test__loop import test_loop
from test_gpus import try_gpu
from load_datasets import load_flowers5_dataset,load_flowers_dataset

train_iter,val_iter = load_flowers_dataset()
net= dense_xgroup_senet(14)
net.load_state_dict(torch.load('model_bestparams_14.pkl'))
test_loop(net,val_iter,device=try_gpu(0),need_confusion=True,num_classes=14)