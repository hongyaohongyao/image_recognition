import argparse
import json
import os
import random
import shutil

import numpy as np
import torch
import torchvision.models as models_torchvision  # networks from torchvision
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets

from svm import SVM
from tqdm import tqdm

import models  # our model
from augmentation import mixup_train
from log import get_logger

parser = argparse.ArgumentParser(description='PyTorch Image Training')

parser.add_argument('--name', default='svm', help='task name')
parser.add_argument('--ds',
                    '--dataset',
                    default='MNIST',
                    help='mnist dataset name')
parser.add_argument('--data-root',
                    default="./datasets",
                    type=str,
                    help='dataset root')
parser.add_argument('-j',
                    '--workers',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (mini_net_sgd: 4)')
parser.add_argument('--mic',
                    '--min-c',
                    default=1,
                    type=float,
                    help='minimal of C')

parser.add_argument('--mac',
                    '--max-c',
                    default=100,
                    type=float,
                    help='maximal of C')

parser.add_argument('--nc',
                    '--num-c',
                    default=20,
                    type=int,
                    help='number of C')
parser.add_argument('-k',
                    '--kernel',
                    default="rbf",
                    type=str,
                    help='larger C mean samller soft margin')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train():
    best_acc = -1
    log_dir = f'run/{args.name}'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "env.json"), "w") as f:
        json.dump(vars(args), f)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    logger = get_logger(log_dir,
                        f'train.log',
                        resume=args.resume,
                        is_rank0=True)

    trans_list = [
        # transforms.RandomAffine(0, (0.1, 0.1)),
        # transforms.RandomRotation((-10, 10)),
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ]
    trans_train = transforms.Compose(trans_list)

    trans_test = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])

    train_set = datasets.__dict__[args.ds](root=args.data_root,
                                           train=True,
                                           download=True,
                                           transform=trans_train)
    train_loader = data.DataLoader(train_set,
                                   batch_size=len(train_set),
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   shuffle=True)

    test_set = datasets.__dict__[args.ds](root=args.data_root,
                                          train=False,
                                          download=True,
                                          transform=trans_test)
    test_loader = data.DataLoader(test_set,
                                  batch_size=len(test_set),
                                  num_workers=args.workers,
                                  pin_memory=True,
                                  shuffle=False)
    
    images, labels = train_loader[0]
    list_c = np.linspace(args.mic, args.mac, args.nc)
    images = images.numpy().reshape(images.shape[0], -1)
    labels = labels.numpy()

    test_loader[0]
    for C in tqdm(list_c):
        model = SVM(kernel=args.kernel, C=args.C)



    

    


if __name__ == '__main__':
    train()