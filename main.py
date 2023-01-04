
import os
import torch
import json
import csv
import numpy as np
from PIL import Image
import torch.nn as nn
import multiprocessing
from models import ResNet18
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from pycocotools.coco import COCO
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


def main(args):
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    
    np.random.seed(args.seed) # Set the random seed of numpy for the data split.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    def change_to_3_channel(x):
        if x.size()[0] == 1:
            return x.repeat(3, 1, 1)
        return x

    train_transform = transforms.Compose([
                                        transforms.Resize(size=(224,224)),
                                        # transforms.RandomCrop(224, padding=4),
                                        # transforms.RandomHorizontalFlip(), 
                                        transforms.ToTensor(),
                                        change_to_3_channel,
                                        transforms.Normalize(args.mean, args.std)
                                    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        change_to_3_channel,
        transforms.Normalize(args.mean, args.std),
    ])

    train_dataset = CocoClassDatasetRandom(images_path = args.val_images_path, annotation_path = val_annotation_path, transform = train_transform)
    val_dataset = CocoClassDatasetRandom(images_path = args.val_images_path, annotation_path = val_annotation_path, transform = val_transform)
    
    print(f'----> number of workers: {args.num_workers}')

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    
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

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr = args.base_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    print(f'Initial random loss: {criterion(outputs, batch_y.squeeze().float())}')