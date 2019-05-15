#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:30:25 2019
@author: skodge
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from vgg import vgg
cuda = True
train_batch_size = 32
test_batch_size = 128

best_loss = float("inf")
best_epoch = -1
best_correct=0
dataset_path = './cifar10'

cuda = cuda and torch.cuda.is_available()
trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
train_mean = trainset.data.mean(axis=(0,1,2))/255  # [0.49139968  0.48215841  0.44653091]
train_std = trainset.data.std(axis=(0,1,2))/255  # [0.24703223  0.24348513  0.26158784]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
    root=dataset_path, train=True, download=True,
    transform=transform_train),
    batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=dataset_path, train=False, download=True,
    transform=transform_test),
    batch_size=test_batch_size, shuffle=False, **kwargs)
    
model = vgg(input_size=3,bit_W=4,bit_A=6,sigma=1.25)
if cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100,150,180], gamma=0.1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            
def test(epoch, best_loss, best_epoch, best_correct, do_quantise,do_add_error,mode,update=False):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model.Inference(data, do_quantise=do_quantise,do_add_error=do_add_error,mode=mode)
        # sum up batch loss
        test_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        if (batch_idx % 100 == 0 and do_add_error==True):
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, batch_idx*test_batch_size+test_batch_size, 100. * correct /
                    (batch_idx*test_batch_size+test_batch_size)))
        

    test_loss /= len(test_loader.dataset)
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
            test_loss, correct, best_correct, 100. * correct /
            len(test_loader.dataset)))
    
    if (best_correct<correct):
        best_epoch = epoch
        best_loss = test_loss
        best_correct=correct
        if (update):
            torch.save(model, "vgg_parameter.pt")
            
    return best_loss, best_epoch, best_correct,correct

epoch=0
model=torch.load("vgg_parameter.pt")
best_loss, best_epoch, best_correct,_ = test(epoch, best_loss, best_epoch, best_correct, do_quantise=False,do_add_error=False,mode=False,update=False)

model=torch.load("vgg_parameter.pt")
model.sigma=0.0
model.bit_A=3
model.bit_W=4
model.quantise_weigths(do_quantise=True)
model.error_initialiser()
best_loss, best_epoch, best_correct,_ = test(epoch, best_loss, best_epoch, best_correct, do_quantise=True,do_add_error=False,mode=True,update=False)

Min=10000
Max=0
MC_correct=torch.zeros(1000)
for i in range(1000):
    correct=0
    model=torch.load("vgg_parameter.pt")
    model.sigma=1.3
    model.bit_A=4
    model.bit_W=4
    model.quantise_weigths(do_quantise=True)
    model.error_initialiser()
    print(i)
    best_loss, best_epoch, _,correct = test(epoch, best_loss, best_epoch, best_correct, do_quantise=True,do_add_error=True,mode=True,update=False)
    MC_correct[i]=correct
    if(correct>Max):
        Max=correct
    if(correct<Min):
        Min=correct
print(Min,Max)
torch.save(MC_correct, "error_accuracy.pt")
