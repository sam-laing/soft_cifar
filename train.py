from resnet import make_resnet_cifar
from read_loader import make_reader, make_loaders  
from evaluate_metrics import evaluate_model

from wrappers.duq_wrapper import DUQWrapper, DUQHead
from wrappers.mahalanobis_wrapper import MahalanobisWrapper
from wrappers.sngp_wrapper import SNGPWrapper, LaplaceRandomFeatureCovariance, LinearSpectralNormalizer, Conv2dSpectralNormalizer, SpectralNormalizedBatchNorm2d


from lr_warmup import WarmupMultiStepLR
from evaluate_metrics import EpochLossCounter

from losses.duq_loss import DUQLoss

import argparse  
from model_structure import create_wrapped_model, set_seed
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import logging

import wandb
from wandbkey import KEY
import warnings







set_seed(99)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--unc_method', default = "sngp", type=str, required=True)
parser.add_argument('--seed', default=99, type=int, help='seed for randomness')
parser.add_argument('--dropout', default=0, type=float, help='dropout rate')

parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--depth', default=20, type=int, help='depth of the model')
parser.add_argument('--gamma', default=0.15, type=float, help='learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='weight decay')
parser.add_argument('--hard', type=bool, default=True)

# SNGPWrapper arguments
parser.add_argument('--is_spectral_normalized', type=bool, default=True)
parser.add_argument('--use_tight_norm_for_pointwise_convs', type=bool, default=True)
parser.add_argument('--spectral_normalization_iteration', type=int, default=1)
parser.add_argument('--spectral_normalization_bound', type=float, default=3)
parser.add_argument('--is_batch_norm_spectral_normalized', type=bool, default=True)
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.getLogger("train")

def main():

    model = make_resnet_cifar(depth=args.depth).to(device)
    reader = make_reader("/home/slaing/ML/2nd_year/sem2/research/CIFAR10H")
    try:
        train_loader, val_loader, test_loader = make_loaders(
                                                            reader, batch_size = 64, 
                                                            split_ratio=[0.8, 0.05, 0.15], 
                                                            use_hard_labels=str2bool(args.hard)
                                                        )
    except Exception as e:
        print(f"Error: {e}")

    wrapped_model = create_wrapped_model(model, args).to(device)

    wandb.login(key=KEY)

    optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    scheduler = WarmupMultiStepLR(optimizer, warmup_epochs=5, milestones=[10, 20, 30], gamma=0.5)
    
    if args.unc_method == "duq":
        criterion = DUQLoss()
    
    with wandb.init(project=f"CIFAR 10 {args.unc_method}", 
                    name=f"{args.unc_method}, lr={args.lr}, BS={args.batch_size}, epochs={args.epochs}, depth={args.depth}, dropout={args.dropout}, hard={args.hard}.pth",
                    config=args) as run:



        for epoch in range(args.epochs):
            optimizer.zero_grad()

            warnings.filterwarnings("ignore")

            train_single_epoch(wrapped_model, train_loader, val_loader, test_loader, optimizer, criterion, epoch)
            scheduler.step(epoch)

            warnings.simplefilter("default")  # Change the filter in this process

    
        




    model_metrics = evaluate_model(wrapped_model, test_loader, device)

    # here one should log the metrics to directory
    # better for main not to return anything
    optimizer.zero_grad()



def train_single_epoch(model, train_loader, val_loader, test_loader, 
                       optimizer, criterion, epoch, device = device, use_mixup=False):
    """   
    Actions: 
        train a single epoch of the model 
    """

    wandb.watch(model, criterion, log="all", log_freq=10)
    model.train()
    optimizer.zero_grad()

    if isinstance(model, SNGPWrapper) and args.gp_cov_momentum < 0:
        model.classifier[-1].reset_covariance_matrix()
    

    epoch_loss = 0
    for idx, (x,y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)

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


    val_loss, val_accuracy = validate(model, val_loader, criterion)
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(
        f"Epoch {epoch}, Loss: {avg_epoch_loss},"
        f"Val Loss: {val_loss}, Val Accuracy: {val_accuracy},"
        f"LR: {optimizer.param_groups[0]['lr']}"
    )
    
    # could also do some wandb logging
    wandb.log({ 
        "train_loss": avg_epoch_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })




def validate(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            if args.unc_method == "sngp":
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
    main()

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_resnet_cifar(depth=args.depth).to(device)
    reader = make_reader("/home/slaing/ML/2nd_year/sem2/research/CIFAR10H")
    try:
        train_loader, val_loader, test_loader = make_loaders(
                                                            reader, batch_size = 64, 
                                                            split_ratio=[0.8, 0.05, 0.15], 
                                                            use_hard_labels=str2bool(args.hard)
                                                        )
    except Exception as e:
        print(f"Error: {e}")
    
    wrapped_model = create_wrapped_model(model, args).to(device)

    loss, acc = validate(wrapped_model, test_loader, nn.CrossEntropyLoss())
    print(loss, acc)
    """


