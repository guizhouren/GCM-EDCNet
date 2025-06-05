import torch
from torch import nn
from train__test__loop import train_loop,test_loop
from load_datasets import load_flowers_dataset,load_flowers5_dataset
from torch import optim
from mobilenetv2 import MobileNetV2
from shuffletnetv1 import ShuffleNet
from Xception import Xception
from mobilenetv3 import mobilenet_v3_small,mobilenet_v3_large
from torchscan import summary
from efficientnetv1 import efficientnet_b0
from vision_transform import vit_base_patch16_224
from efficientnetv2 import efficientnetv2_s
from xgroup_dr import XGroupNet
from DenseNet import dense_net
from ResNet import res_net
from Densenet_121 import DenseNet121
from DenseNet_xgroupnet import dense_xgroupnet
from DenseNet_xgroup_ecanet import dense_xgroup_ecanet
from test_gpus import try_gpu
from DenseNet_xgroup_senet import dense_xgroup_senet
from DenseNet_xgroup_cbam import dense_xgroup_cbam
from repvgg import create_RepVGG_A0
from swinTransformer import SwinTransformer
from DenseNet_xgroupnet_withoutshuffle import dense_xgroupnet_withoutshuffle
from DenseNet_xgroupnet_withoutdense import dense_xgroupnet_withoutdense
from convnext import convnext_tiny,convnext_small,convnext_large
from mobilevit import mobile_vit_x_small
from modelCNN import ModelCNN
from fasternet import FasterNetT0
from ragnet import create_regnet
#net = vit_base_patch16_224(num_classes=5)
#net = dense_xgroupnet_withoutdense(5)
#net = ShuffleNet(3,[240,480,960],class_num=5)
#net = Xception(num_classes=5)
#net = create_RepVGG_A0(num_class=5)
#net = create_regnet(model_name="regnetx_600mf",num_classes=5)
net = mobilenet_v3_large(num_classes=5)
print(summary(net,(3,224,224)))

train_iter,val_iter = load_flowers5_dataset()
#train_iter,val_iter = load_flowers5_dataset()
print(len(train_iter),len(val_iter))


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=1e-3)


train_loop(net,loss_fn,optimizer,train_iter,val_iter,100,device=try_gpu(1),acc=0,cl_type=5)
test_loop(net,val_iter,device=try_gpu(1),need_confusion=True,num_classes=5)
print('------------------------------------------------------')
net.load_state_dict(torch.load('model_bestparams_5.pkl'))
test_loop(net,val_iter,device=try_gpu(1),need_confusion=True,num_classes=5)