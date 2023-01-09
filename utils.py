
import torch
import argparse
import numpy as np
from collections import ChainMap
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

def fix_random_seeds(seed: int = 31):
    """Fix random seed.
    Args:
        seed: intial random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class DotDict(dict):
    """Dot notation access to dictionary attributes.
    Source: https://stackoverflow.com/a/23689767
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore

def scores(y_true, y_pred):
    AP = average_precision_score(y_true, y_pred, average='samples')
    f1 = f1_score(y_true, y_pred, average='samples')
    return AP, f1

def checkpoint(model, acc, epoch, outModelName):
    # Save checkpoint.
    print('Saving..')
    state = {
        'state_dict': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpointPretrain'):
        os.mkdir('checkpointPretrain')
    torch.save(state, f'./checkpointPretrain/{outModelName}.t7')

def adjust_learning_rate(optimizer, base_learning_rate, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_arguments(parser: argparse.ArgumentParser, default_args: argparse.Namespace) -> ChainMap:
    """Parse arguments from the command line.
    Args:
        parser: command line parser.
        default_args: default parse arguments specified in config.py
    Returns:
        args_col: command line argument.
    """
    parser.add_argument("--seed", type=int, 
            default=1, help="seed"
    )
    parser.add_argument(
        "--epochs",
        default=default_args.epochs,
        type=int,
        help="no. of epochs for training",
    )
    parser.add_argument(
        "--num_classes",
        default=default_args.num_classes,
        type=int,
        help="no. of classes for the task",
    )
    parser.add_argument(
        "--batch_size",
        default=default_args.batch_size,
        type=int,
        help="Batch size",
    )

    args = parser.parse_args()
    fix_random_seeds(args.seed)
    args_col = ChainMap(vars(args), vars(default_args))

    return args_col