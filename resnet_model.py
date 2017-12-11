from __future__ import print_function, division

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
from data.labels import get_labels

plt.ion()   # interactive mode



# Data augmentation and normalization for training
# Just normalization for validation
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
data_transforms = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

data_dir = 'data'

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
    for x in ['training', 'validation']
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=128, shuffle=True, num_workers=4
    )
    for x in ['training', 'validation']
}

dataset_sizes = { x: len(image_datasets[x]) for x in ['training', 'validation']}
# print(dataset_sizes)
class_names = image_datasets['training'].classes
# print(image_datasets['training'].classes == image_datasets['validation'].classes)
# print(len(image_datasets['training'].classes), len(image_datasets['validation'].classes))

use_gpu = torch.cuda.is_available()

def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    axis.imshow(inp)
    out_png = './out_file.png'
    plt.savefig(out_png, dpi=150)

img, label = next(iter(dataloaders['training']))
print(img.size(), label.size())
fig = plt.figure(1, figsize=(16, 4))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.05)
for i in range(4):
    ax = grid[i]
    imshow(ax, img[i])


def visualize_model(dataloaders, model, num_images=16):
    cnt = 0
    fig = plt.figure(1, figsize=(16, 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.05)
    for i, (inputs, labels) in enumerate(dataloaders['validation']):
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            ax = grid[cnt]
            imshow(ax, inputs.cpu().data[j])
            ax.text(10, 210, '{}/{}'.format(preds[j], labels.data[j]),
                    color='k', backgroundcolor='w', alpha=0.8)
            cnt += 1
            if cnt == num_images:
                return
    plt.savefig('./out_file_2.png', dpi=150)

def train_model(dataloaders, model, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {'training': len(dataloaders['training'].dataset),
                     'validation': len(dataloaders['validation'].dataset)}

    for epoch in range(num_epochs):
        for phase in ['training', 'validation']:
            if phase == 'training':
                # scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'training':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'training':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'validation' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, train_epoch_acc,
                valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

class TestImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        identifier = path.split('/')[-1].split('.')[0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, identifier




resnet = models.resnet152(pretrained=True)
# freeze all model parameters
for param in resnet.parameters():
    param.requires_grad = False

num_features = resnet.fc.in_features

fc_layers = nn.Sequential(
                nn.Linear(num_features, 4096),
                nn.ELU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(4096, 120),
                nn.ELU(inplace=True),
                nn.Dropout(p=0.1),
            )
resnet.fc = torch.nn.Linear(num_features, 120)
print(resnet)
if use_gpu:
    resnet = resnet.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

start_time = time.time()
model = torch.load('resnet_model.pth')
model = train_model(dataloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
torch.save(model, 'resnet_model.pth')

print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

# visualize_model(dataloaders, resnet)
