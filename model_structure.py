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
    if wrapper != "basic":
        assert args.dropout == 0, "Cannot have dropout with other wrappers"

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
            num_mc_samples=10
        )
    else:
        raise ValueError(f"method {wrapper} not found")

    return wrapped_model







