
import os
import torch
import json
import csv
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import multiprocessing
from model import ResNet18
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from config import args as default_args
from utils import DotDict, parse_arguments
import torchvision.transforms as transforms
from dataloader import CocoClassDatasetRandom


def main(args):

    def change_to_3_channel(x):
        if x.size()[0] == 1:
            return x.repeat(3, 1, 1)
        return x

    train_transform = transforms.Compose([
                                        # transforms.Resize(size=(224,224)),
                                        # transforms.RandomCrop(224, padding=4),
                                        # transforms.RandomHorizontalFlip(), 
                                        transforms.ToTensor(),
                                        change_to_3_channel,
                                        transforms.Normalize(args.mean, args.std)
                                    ])

    val_transform = transforms.Compose([
        # transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        change_to_3_channel,
        transforms.Normalize(args.mean, args.std),
    ])

    train_dataset = CocoClassDatasetRandom(images_path = args.train_images_path, annotation_path = args.train_annotation_path, transform = train_transform)
    val_dataset = CocoClassDatasetRandom(images_path = args.val_images_path, annotation_path = args.val_annotation_path, transform = val_transform)
    
    print(f'----> number of workers: {args.num_workers}')

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    net = ResNet18() 
    inp_features = net.linear.in_features
    net.linear = torch.nn.Linear(in_features=inp_features, out_features=args.num_classes)
    
    batch_x, batch_y = next(iter(trainloader))

    print('-----> verify if model is run on random data')

    outputs = net(batch_x)
    
    print(f'Outputs shape: {outputs.shape}, batch labels shape: {batch_y.shape}') 

    result_folder = './resultsPretrain/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    logname = result_folder + net.__class__.__name__ + '_pretrain' + '.csv'


    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr = args.base_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    print(f'Initial random loss: {criterion(outputs, batch_y.squeeze().float())}')
    
    predicted = torch.round(torch.sigmoid(outputs))
    random_acc = predicted.eq(batch_y.squeeze().float()).cpu().sum()/(args.batch_size * args.num_classes) * 100
    print(f'Initial random acc: {random_acc} %')
    
    for epoch in range(args.num_epochs):
        print('---- starting epoch: {epoch} -----')
        
        train_loss, train_acc, train_AP, train_f1 = train(net, criterion, optimizer, trainloader, epoch, use_cuda)
        test_loss, test_acc, test_AP, test_f1 = test(net, criterion, testloader, epoch, args.outModelName)
        
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser, default_args)
    
    args = DotDict(args)
    
    main(args)
    