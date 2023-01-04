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
    lr = 0.001,
    momentum = 0.9,
    base_learning_rate = 0.1,
    weight_decay = 1e-4,
    outModelName = 'randomModel',
    resume = False,
    device = 'cuda',
    
    val_images_path = './coco/images/val2017'
    val_annotation_path = './coco/annotations/instances_val2017.json'
    pretrained = '/content/drive/MyDrive/PhD_Research/Transfer_learning_with_correlations/PhD_research/checkpoint/', 
    dump_path = '/content/drive/MyDrive/PhD_Research/Transfer_learning_with_correlations/PhD_research/checkpoint',
    
    # mean and std stats
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    
    ## hyper-parameters
    best_acc = 0, 
    start_epoch = 0, 
    batch_size = 32, 
    max_epochs = 10, 
    max_epochs_target = 10, 
    base_learning_rate = 0.1, 
    )

args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()


