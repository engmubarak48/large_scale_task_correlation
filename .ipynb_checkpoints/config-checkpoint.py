from argparse import Namespace
import multiprocessing
from typing import no_type_check
import torch
import numpy as np
seed = 0
np.random.seed(seed) # Set the random seed of numpy for the data split.
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

args = Namespace(
    device = 'cuda',
    
    val_images_path = './coco_dataset/images/val2017',
    val_annotation_path = './coco_dataset/annotations/instances_val2017.json',
    train_images_path = './coco_dataset/images/train2017',
    train_annotation_path = './coco_dataset/annotations/instances_train2017.json',
    pretrained = './checkpoint', 
    dump_path = './checkpoint',
    resultPath = './resultPath',
    # mean and std stats
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    
    ## hyper-parameters
    num_classes = 40,
    best_acc = 0, 
    start_epoch = 0, 
    batch_size = 512, 
    epochs = 2, 
    lr = 0.001,
    momentum = 0.9,
    weight_decay = 1e-4,
    outModelName = 'randomModel',
    resume = False,
    base_learning_rate = 0.1, 
    )

args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()


