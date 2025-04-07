from resnet import make_resnet_cifar
from data import make_reader, make_loaders 
from mixup import mixup_datapoints, cutmix_datapoints 
from evaluate_metrics import evaluate_model

from wrappers.duq_wrapper import DUQWrapper, DUQHead
from wrappers.mahalanobis_wrapper import MahalanobisWrapper
from wrappers.sngp_wrapper import SNGPWrapper, LaplaceRandomFeatureCovariance, LinearSpectralNormalizer, Conv2dSpectralNormalizer, SpectralNormalizedBatchNorm2d
from wrappers.dropout_wrapper import DropoutWrapper

from lr_warmup import WarmupMultiStepLR
from evaluate_metrics import EpochLossCounter

from losses.duq_loss import DUQLoss

import argparse  
from model_structure import create_wrapped_model
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import logging
import os
import json

import wandb
from wandbkey import KEY  # wandbkey.py is private for security reasons, make your own to use wandb
import warnings

from utils import str2bool, calc_gradient_penalty, train_single_epoch, validate, set_seed


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--unc_method', default = "basic", type=str)
parser.add_argument('--seed', default=43, type=int, help='seed for randomness')
parser.add_argument('--dropout', default=0, type=float, help='dropout rate')

parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
parser.add_argument('--depth', default=20, type=int, help='depth of the model')
parser.add_argument('--gamma', default=0.5, type=float, help='learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='weight decay')
parser.add_argument('--do_augmentation', default=True, type=bool, help='whether to do data augmentation')
parser.add_argument('--hard', type=bool, default=True)
parser.add_argument('--mixup', type=float, default=0.0)
parser.add_argument('--mixup_prob', type=float, default=0.0)
parser.add_argument('--cutmix', type=float, default=0.0)
parser.add_argument('--cutmix_prob', type=float, default=0.0)

# SNGPWrapper arguments
parser.add_argument('--is_spectral_normalized', type=bool, default=True)
parser.add_argument('--use_tight_norm_for_pointwise_convs', type=bool, default=True)
parser.add_argument('--spectral_normalization_iteration', type=int, default=1)
parser.add_argument('--spectral_normalization_bound', type=float, default=3)
parser.add_argument('--is_batch_norm_spectral_normalized', type=bool, default=False)
parser.add_argument('--num_mc_samples', type=int, default=1000)
parser.add_argument('--num_random_features', type=int, default=1024)
parser.add_argument('--gp_kernel_scale', type=float, default=1.0)
parser.add_argument('--gp_output_bias', type=float, default=0.0)
parser.add_argument('--gp_random_feature_type', type=str, default="orf")
parser.add_argument('--is_gp_input_normalized', type=bool, default=False)
parser.add_argument('--gp_cov_momentum', type=float, default=-1)
parser.add_argument('--gp_cov_ridge_penalty', type=float, default=1)
parser.add_argument('--gp_input_dim', type=int, default=-1)


# MahalanobisWrapper arguments
parser.add_argument('--magnitude', type=float, default=0.1)
parser.add_argument('--weight_path', type=str, default="/home/slaing/ML/2nd_year/sem2/research/models/new_net/536411_hard_False_BS_64_LR_0.04_epochs_100_depth_32_gamma_0.15_mom_0.9.pth")

# DUQWrapper arguments
parser.add_argument('--num_hidden_features', type=int, default=1024)
parser.add_argument('--rbf_length_scale', type=float, default=0.1)
parser.add_argument('--ema_momentum', type=float, default=0.999)
parser.add_argument('--gradient_penalty_weight', type=float, default=0.1)


args = parser.parse_args()


def main(device, reader, args=args):
    set_seed(args.seed)
    model = make_resnet_cifar(depth=args.depth).to(device)
    try:
        train_loader, val_loader, test_loader = make_loaders(
                                                            reader, batch_size = 64, 
                                                            split_ratio=[0.8, 0.05, 0.15], 
                                                            use_hard_labels=str2bool(args.hard), 
                                                            do_augmentation=str2bool(args.do_augmentation)
                                                        )
    except Exception as e:
        print(f"Error: {e}")

    wrapped_model = create_wrapped_model(model, args).to(device)
    assert not (args.mixup > 0 and args.cutmix > 0), "Cannot use both mixup and cutmix at the same time"
    wandb.login(key=KEY)

    optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    scheduler = WarmupMultiStepLR(optimizer, warmup_epochs=5, milestones=[50,100,150,190,210,230], gamma=args.gamma)
    
    if args.unc_method == "duq":
        criterion = DUQLoss()

    job_id = os.environ.get("SLURM_JOB_ID")

    common_hp_str = f"{job_id}_hard={args.hard}, aug={args.do_augmentation},dropout={args.dropout}, mixup={args.mixup}:{args.mixup_prob}, cutmix={args.cutmix}:{args.cutmix_prob}"
    if args.unc_method == "basic":
        name = "basic, " + common_hp_str + ".pth"
    elif args.unc_method == "sngp":
        name = "sngp, " + common_hp_str + ".pth"
    elif args.unc_method == "mahalanobis":
        name = "mahalanobis, " + f"magnitude={args.magnitude}, weight_path={args.weight_path}" + common_hp_str + ".pth"
    elif args.unc_method == "duq":
        name = "duq, " + common_hp_str + ".pth"
    else:
        raise ValueError(f"Unknown uncertainty method: {args.unc_method}")
 
    with wandb.init(
        project=f"CIFAR10 soft 10 ", 
        name=name,
        tags = [
            args.unc_method,
            f"hard={args.hard}",
            f"aug={args.do_augmentation}",
            f"dropout={args.dropout}",
            f"mixup={args.mixup}:{args.mixup_prob}",
            f"cutmix={args.cutmix}:{args.cutmix_prob}",
            f"depth={args.depth}",
            f"lr={args.lr}",
            f"epochs={args.epochs}",
            f"batch_size={args.batch_size}",
            f"gamma={args.gamma}",
            f"momentum={args.momentum}",
            f"weight_decay={args.weight_decay}",
            f"num_mc_samples={args.num_mc_samples}",
            f"num_random_features={args.num_random_features}"
        ],
        config=args
        ) as run:

        wandb.watch(model, criterion, log="all", log_freq=10)
        for epoch in range(args.epochs):
            optimizer.zero_grad()

            warnings.filterwarnings("ignore")

            train_single_epoch(wrapped_model, train_loader, val_loader, test_loader, optimizer, criterion, epoch, device)
            scheduler.step(epoch)

            warnings.simplefilter("default")  # Change the filter in this process

        model_metrics = evaluate_model(wrapped_model, test_loader, device)
        model_metrics = {k: float(v) for k,v in model_metrics.items()}
        # save the model 
        path = "/mnt/qb/work/oh/owl886/soft_cifar/models/"
        torch.save(wrapped_model.state_dict(), path + f"{run.name}.pth")

        #save the json
        with open(path + f"{run.name}.json", "w") as f:
            json.dump(model_metrics, f)


        
