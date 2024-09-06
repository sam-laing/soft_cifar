from train import main 

from read_loader import make_reader
import torch

import argparse 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--unc_method', default = "sngp", type=str, required=True)
parser.add_argument('--seed', default=42, type=int, help='seed for randomness')
parser.add_argument('--dropout', default=0, type=float, help='dropout rate')

parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--epochs', default=250, type=int, help='number of total epochs to run')
parser.add_argument('--depth', default=20, type=int, help='depth of the model')
parser.add_argument('--gamma', default=0.15, type=float, help='learning rate decay')
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

reader = make_reader("/mnt/qb/work/oh/owl886/datasets/CIFAR10H")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main(args, reader, device)