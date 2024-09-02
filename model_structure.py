import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import wandb
import os


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from read_loader import make_datasets, make_loaders, make_reader
from resnet import make_resnet_cifar
from wandbkey import KEY

import random

from wrappers.sngp_wrapper import SNGPWrapper
from wrappers.mahalanobis_wrapper import MahalanobisWrapper
from wrappers.duq_wrapper import DUQWrapper
from wrappers.dropout_wrapper import DropoutWrapper
from wrappers.model_wrapper import ModelWrapper

import logging  
import sys

def create_wrapped_model(model, args):
    """    
    Inputs: 
        model: nn.Module - the model to be wrapped 
        args: argparse.Namespace - the arguments from the command line which are the HPs of the wrappers
    Outputs:
        wrapped_model: nn.Module - the wrapped model
    """
    wrapper = args.unc_method

    if wrapper == "sngp":
        wrapped_model = SNGPWrapper(
            model=model,
            is_spectral_normalized=args.is_spectral_normalized,
            use_tight_norm_for_pointwise_convs=args.use_tight_norm_for_pointwise_convs,
            spectral_normalization_iteration=args.spectral_normalization_iteration,
            spectral_normalization_bound=args.spectral_normalization_bound,
            is_batch_norm_spectral_normalized=args.is_batch_norm_spectral_normalized,
            num_mc_samples=args.num_mc_samples,
            num_random_features=args.num_random_features,
            gp_kernel_scale=args.gp_kernel_scale,
            gp_output_bias=args.gp_output_bias,
            gp_random_feature_type=args.gp_random_feature_type,
            is_gp_input_normalized=args.is_gp_input_normalized,
            gp_cov_momentum=args.gp_cov_momentum,
            gp_cov_ridge_penalty=args.gp_cov_ridge_penalty,
            gp_input_dim=args.gp_input_dim
        )
    elif wrapper == "mahalanobis":
        wrapped_model = MahalanobisWrapper(
            model,
            magnitude=args.magnitude,
            weight_path=args.weight_path
        )
    elif wrapper == "duq":
        wrapped_model = DUQWrapper(
            model,
            num_hidden_features=args.num_hidden_features,
            rbf_length_scale=args.rbf_length_scale,
            ema_momentum=args.ema_momentum
        )
    elif (wrapper=="basic") and (args.dropout==0):
        wrapped_model = ModelWrapper(model)
    
    elif (wrapper=="basic") and (args.dropout > 0):
        wrapped_model = DropoutWrapper(
            model=model,
            dropout_probability=args.dropout,
            is_filterwise_dropout=False,
            num_mc_samples=5
        )
    else:
        raise ValueError(f"method {wrapper} not found")

    return wrapped_model






def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--unc_method', default = "sngp", type=str, required=True)

    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--depth', default=20, type=int, help='depth of the model')
    parser.add_argument('--gamma', default=0.15, type=float, help='learning rate decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--hard', type=bool, default=False)

    # SNGPWrapper arguments
    parser.add_argument('--is_spectral_normalized', type=bool, default=True)
    parser.add_argument('--use_tight_norm_for_pointwise_convs', type=bool, default=False)
    parser.add_argument('--spectral_normalization_iteration', type=int, default=1)
    parser.add_argument('--spectral_normalization_bound', type=float, default=0.95)
    parser.add_argument('--is_batch_norm_spectral_normalized', type=bool, default=True)
    parser.add_argument('--num_mc_samples', type=int, default=10)
    parser.add_argument('--num_random_features', type=int, default=128)
    parser.add_argument('--gp_kernel_scale', type=float, default=1.0)
    parser.add_argument('--gp_output_bias', type=float, default=0.0)
    parser.add_argument('--gp_random_feature_type', type=str, default="orf")
    parser.add_argument('--is_gp_input_normalized', type=bool, default=False)
    parser.add_argument('--gp_cov_momentum', type=float, default=0.999)
    parser.add_argument('--gp_cov_ridge_penalty', type=float, default=1e-3)
    parser.add_argument('--gp_input_dim', type=int, default=128)

    # MahalanobisWrapper arguments
    parser.add_argument('--magnitude', type=float, default=0.1)
    parser.add_argument('--weight_path', type=str, default="/home/slaing/ML/2nd_year/sem2/research/models/new_net/536411_hard_False_BS_64_LR_0.04_epochs_100_depth_32_gamma_0.15_mom_0.9.pth")

    # DUQWrapper arguments
    parser.add_argument('--num_hidden_features', type=int, default=1024)
    parser.add_argument('--rbf_length_scale', type=float, default=0.1)
    parser.add_argument('--ema_momentum', type=float, default=0.999)
    parser.add_argument('--gradient_penalty_weight', type=float, default=0.1)


    args = parser.parse_args()

    reader = make_reader("/home/slaing/ML/2nd_year/sem2/research/CIFAR10H")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_loaders(reader, use_hard_labels=True, batch_size=128, split_ratio=[0.8, 0.05, 0.15])

    model = make_resnet_cifar(depth=args.depth).to(device)

    wrapped_model = create_wrapped_model(model, args)
    wrapped_model.training = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(wrapped_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=args.gamma)

    for x,y in test_loader:
        print(wrapped_model(x), wrapped_model(x).shape)
        break
