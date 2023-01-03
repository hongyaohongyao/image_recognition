import argparse
import json
import os
import random
import shutil
from datetime import datetime
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
import pickle as pkl
import models  # our model
from augmentation import mixup_train
from log import get_logger
from sklearn.svm import SVC

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
parser.add_argument('-C', default=1, type=float, help='C')
parser.add_argument('--sample-rate',
                    '--sr',
                    default=0.5,
                    type=float,
                    help='sample train data')
parser.add_argument('--nc', '--num-c', default=5, type=int, help='number of C')
parser.add_argument('-k',
                    '--kernel',
                    default="rbf",
                    type=str,
                    help='larger C mean samller soft margin')
parser.add_argument('--seed',
                    default=31,
                    type=int,
                    help='seed for initializing training. ')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train():
    best_acc = -1
    log_dir = f'run_svm/{args.name}'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "env.json"), "w") as f:
        json.dump(vars(args), f)
    logger = get_logger(log_dir, f'train.log', resume="", is_rank0=True)

    trans_list = [
        # transforms.RandomAffine(0, (0.1, 0.1)),
        # transforms.RandomRotation((-10, 10)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ]
    trans_train = transforms.Compose(trans_list)

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])

    train_set = datasets.__dict__[args.ds](root=args.data_root,
                                           train=True,
                                           download=True,
                                           transform=trans_train)
    train_loader = data.DataLoader(train_set,
                                   batch_size=int(
                                       len(train_set) * args.sample_rate),
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

    X, y = list(train_loader)[0]
    X = X.numpy().reshape(X.shape[0], -1)
    y = y.numpy()

    X_test, y_test = list(test_loader)[0]
    X_test = X_test.numpy().reshape(X_test.shape[0], -1)
    y_test = y_test.numpy()

    logger.info(f'start train {datetime.now()}, C: {args.C}')
    model = SVM(kernel=args.kernel, C=args.C)
    # model = SVC(kernel=args.kernel, C=c)
    model.fit(X, y)
    pred = model.predict(X)
    train_acc = (pred == y).mean()
    logger.info('Train/acc %.5f' % (train_acc))

    logger.info(f'start test, {datetime.now()}')
    test_pred = model.predict(X_test)
    test_acc = (test_pred == y_test).mean()
    logger.info('Test/acc %.5f' % (test_acc))
    # save
    save_dir = os.path.join(log_dir, 'model.pkl')
    with open(save_dir, 'wb') as f:
        pkl.dump(
            {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "C": args.C,
                "model": model
            }, f)


if __name__ == '__main__':
    setup_seed(args.seed)
    train()