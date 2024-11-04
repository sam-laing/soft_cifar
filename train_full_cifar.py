from resnet import make_resnet_cifar
from read_loader import make_reader, make_loaders
from cifar10 import make_larger_hard_loaders 
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
from model_structure import create_wrapped_model, set_seed
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

import logging
import os
import json

import wandb
from wandbkey import KEY
import warnings


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--unc_method', default = "basic", type=str)
parser.add_argument('--seed', default=999, type=int, help='seed for randomness')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--depth', default=20, type=int, help='depth of the model')
parser.add_argument('--gamma', default=0.4, type=float, help='learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--do_augmentation', default=True, type=bool, help='whether to do data augmentation')
parser.add_argument('--hard', type=bool, default=True)
parser.add_argument('--mixup', type=float, default=0.0)
parser.add_argument('--mixup_prob', type=float, default=0.15)
parser.add_argument('--cutmix', type=float, default=0.0)
parser.add_argument('--cutmix_prob', type=float, default=0.15)

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

def main(args, device):
    set_seed(args.seed)
    model = make_resnet_cifar(depth=args.depth).to(device)
    train_loader, val_loader, test_loader = make_larger_hard_loaders(
        path = "/mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py/", 
        split_ratio=[0.8, 0.05, 0.15],
        batch_size=args.batch_size, 
        do_augmentation=args.do_augmentation,
        seed=args.seed
    )
        


    
    full_size = 0
    for x,y in train_loader:
        full_size += len(x)
    print(f"Full size of the dataset: {full_size}")


    wrapped_model = create_wrapped_model(model, args).to(device)
    assert not (args.mixup > 0 and args.cutmix > 0), "Cannot use both mixup and cutmix at the same time"
    wandb.login(key=KEY)

    optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    # scheduler should be normal multi step with milestones and gamma = 0.5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65, 120, 165], gamma=args.gamma)
     
    if args.unc_method == "duq":
        criterion = DUQLoss()

    job_id = os.environ.get("SLURM_JOB_ID")

    common_hp_str = f"{job_id}_hard={args.hard}, aug={args.do_augmentation},dropout={args.dropout}, mixup={args.mixup}, cutmix={args.cutmix}"
    if args.unc_method == "basic":
        name = "basic, " + common_hp_str 
    elif args.unc_method == "sngp":
        name = "sngp, " + common_hp_str 
    elif args.unc_method == "mahalanobis":
        name = "mahalanobis, " + f"magnitude={args.magnitude}, weight_path={args.weight_path}" + common_hp_str 
    elif args.unc_method == "duq":
        name = "duq, " + common_hp_str 
    else:
        raise ValueError(f"Unknown uncertainty method: {args.unc_method}")
 
    with wandb.init(project=f"CIFAR10 full {args.unc_method}, seed {args.seed}", 
                    name=name,
                    config=args) as run:


        wandb.watch(model, criterion, log="all", log_freq=10)
        for epoch in range(args.epochs):
            optimizer.zero_grad()

            warnings.filterwarnings("ignore")


            train_single_epoch(wrapped_model, train_loader, val_loader, test_loader, optimizer, criterion, epoch, device)
            scheduler.step(epoch)


        model_metrics = evaluate_model(wrapped_model, test_loader, device)
        model_metrics = {k: float(v) for k,v in model_metrics.items()}
        # save the model 
        path = "/mnt/qb/work/oh/owl886/soft_cifar/models/"
        torch.save(wrapped_model.state_dict(), path + f"{run.name}.pth")

        #save the json
        with open(path + str(epoch) + f"{run.name}.json", "w") as f:
            json.dump(model_metrics, f)


        
def train_single_epoch(model, train_loader, val_loader, test_loader, 
                       optimizer, criterion, epoch, device, use_mixup=False):
    """   
    Actions: 
        train a single epoch of the model        do_avg = True
    """

    model.train()
    optimizer.zero_grad()

    if isinstance(model, SNGPWrapper) and args.gp_cov_momentum < 0:
        model.classifier[-1].reset_covariance_matrix()
    

    epoch_loss = 0
    for idx, (x,y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        r = torch.rand(1).item()
        if args.mixup > 0 and r < args.mixup_prob:
            x, y = mixup_datapoints(x, y, device, alpha=args.mixup)
        if args.cutmix > 0 and r < args.cutmix_prob:
            x, y = cutmix_datapoints(x, y, device, alpha=args.cutmix)

        if isinstance(model, DUQWrapper):
            x.requires_grad_(True)

        def forward():
            output = model(x)
            loss = criterion(output, y)

            if isinstance(model, DUQWrapper):
                gradient_penalty = calc_gradient_penalty(x, output)
                loss += args.gradient_penalty_weight * gradient_penalty
            
            return loss

        def backward(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if isinstance(model, DUQWrapper):
                x.requires_grad_(False)

                with torch.no_grad():
                    model.eval()
                    model.update_centroids(x,y)
                    model.train()
        
        loss = forward()
        epoch_loss += loss.item()
        backward(loss)


    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(
        f"Epoch {epoch}, Loss: {avg_epoch_loss},"
        f"Val Loss: {val_loss}, Val Accuracy: {val_accuracy},"
        f"LR: {optimizer.param_groups[0]['lr']}"
    )
    lr = optimizer.param_groups[0]["lr"]
    # could also do some wandb logging

    wandb.log({ 
        "train_loss": avg_epoch_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy, 
        "lr": lr
    })  




def validate(model, val_loader, criterion, device):
    if len(val_loader) == 0:
        return 0, 0

    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            if isinstance(model, SNGPWrapper) or isinstance(model, DropoutWrapper):
                # take mean along 1 dim
                outputs = model(images)["logit"].mean(1).to(device)

            else:
                outputs = model(images)["logit"].squeeze(1).to(device)

            val_loss += criterion(outputs, labels)
            total += labels.size(0)
            correct += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        return val_loss, val_accuracy 


def calc_gradients_input(x, pred):
    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=x,
        grad_outputs=torch.ones_like(pred),
        retain_graph=True,  # Graph still needed for loss backprop
    )[0]

    gradients = gradients.flatten(start_dim=1)

    return gradients


def calc_gradient_penalty(x, pred):
    gradients = calc_gradients_input(x, pred)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two-sided penalty
    gradient_penalty = (grad_norm - 1).square().mean()

    return gradient_penalty


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', "True", 't', 'y', '1', 1):
        return True
    elif v.lower() in ('no', 'false', "True", 'f', 'n', '0', 0):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args, device)