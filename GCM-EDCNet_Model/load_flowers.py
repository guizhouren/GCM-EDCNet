import torch
import torchvision
from torchvision import datasets,transforms
import os

def load_dataset(data_dir =os.getcwd()+'/MyDatasets/'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.476, 0.475, 0.406), (0.28, 0.269, 0.273))
        ]),
        'val': transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.476, 0.475, 0.406), (0.28, 0.269, 0.273))
        ])
    }
    data_train=datasets.ImageFolder(data_dir+'train',transform=data_transforms['train'])
    data_test=datasets.ImageFolder(data_dir+'val',transform=data_transforms['val'])
    print(len(data_train)),print(len(data_test))

    train_iter = torch.utils.data.DataLoader(data_train,shuffle=True,batch_size=128)
    val_iter = torch.utils.data.DataLoader(data_test,shuffle=True,batch_size=20)
    print(len(train_iter),len(val_iter))
    return train_iter,val_iter

train_iter,val_iter=load_dataset()
