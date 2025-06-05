import torch
import torchvision
from torchvision import transforms
import os

data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }


data_dir = os.getcwd()+'/flowers_5/train'
data_train = torchvision.datasets.ImageFolder(data_dir,transform=data_transforms['train'])
data_stack = torch.stack([img for img,_ in data_train],dim=3)
print(data_stack.shape)

mean=data_stack.view(3,-1).mean(dim=1)
std=data_stack.view(3,-1).std(dim=1)
print(mean)
print(std)