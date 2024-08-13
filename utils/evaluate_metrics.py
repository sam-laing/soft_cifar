import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import sys
import time
import random
from torch import Tensor

import os
import random

import torchvision
from tqdm.auto import tqdm

from ..read_loader import make_reader, make_loaders, make_datasets
from ..resnet import make_resnet_cifar
from ..load_ood import get_svhn_loader

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from collections import OrderedDict

def evaluate_model(model, test_loader, device):
    """    
    given the model and test loader, return a dict of important metrics for the model 
    """

    outputs, labels = get_model_outputs_and_labels(test_loader, model, device)
    ece, oe = get_ece_and_overconf_err(outputs, labels, num_bins=10)

    acc = _get_accuracy(outputs, labels)
    precision, recall = _get_precision_and_recall(outputs, labels)

    critierion = nn.CrossEntropyLoss()
    loss = critierion(outputs, labels)

    # Get the predicted probabilities
    outputs_prob = torch.softmax(outputs, dim=1).cpu().detach().numpy()

    # Convert labels to numpy array
    labels_one_hot = labels.cpu().numpy()

    # Calculate the AUROC for each class and then average
    auroc = roc_auc_score(labels_one_hot, outputs_prob, average='macro', multi_class='ovr')

    brier_score = np.mean(np.sum((outputs_prob - labels_one_hot) ** 2, axis=1))

    #f1 = f1_score(labels_one_hot, np.argmax(outputs_prob, axis=1), average='macro')

    
    return {
        "Accuracy": acc, "ECE": ece, "Overconfidence Error": oe, "Log Loss": loss.item(), "AUROC": auroc, 
        "Precision" : precision, "Recall": recall, "Brier Score": brier_score
    }


def get_model_outputs_and_labels(test_loader: DataLoader, model:nn.Module, device):
  """
  iterate through test loader and return single numpy array of all outputs and labels
  """
  model.eval()
  outputs_list = []
  labels_list = []
  with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      outputs_list.append(outputs.cpu())
      labels_list.append(labels.cpu())

  all_outputs = torch.cat(outputs_list)
  all_outputs = torch.softmax(all_outputs, dim=1)
  all_labels = torch.cat(labels_list)

  return all_outputs, all_labels


def _get_accuracy(outputs, labels):
    return (100*(torch.argmax(labels,1) == torch.argmax(outputs, 1)).sum() / labels.shape[0]).item()

def _get_precision_and_recall(outputs, labels):
    _, predicted_labels = torch.max(outputs, 1)
    true_labels = torch.argmax(labels, 1)

    predicted_labels_np = predicted_labels.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    precision = precision_score(true_labels_np, predicted_labels_np, average='macro')
    recall = recall_score(true_labels_np, predicted_labels_np, average='macro')

    return precision, recall


def _get_bins(outputs: torch.tensor, labels: torch.tensor, n_bins=10):
    '''
    Computes the Expected Calibration Error (ECE).
    outputs: (n_samples, n_classes) already passed through softmax
    labels: (n_samples, n_classes) as indices of the true classes

    returns: bins_pred, bins_y, bin_sizes
    '''
    assert outputs.shape[0] == labels.shape[0]
    N = outputs.shape[0]
    probs = outputs
    labels = torch.argmax(labels, axis=1)
    corrects = (torch.argmax(probs, axis=1) == labels).float()
    corrects = corrects.unsqueeze(1)
    # just need the model's confidence of the true class 
    pred_conf = torch.gather(probs, 1, labels.unsqueeze(1))
    bin_sizes = []
    confs = []
    accs = []
    for i in range(n_bins):
        bin_min = i / n_bins
        bin_max = (i + 1) / n_bins

        B = torch.where((pred_conf > bin_min) & (pred_conf <= bin_max))
        size = B[0].shape[0]
        bin_sizes.append(size)
        confs.append(torch.mean(pred_conf[B]).item())
        accs.append(torch.mean(corrects[B]).item())

    return confs, accs, bin_sizes


def get_ece_and_overconf_err(outputs, labels, num_bins = 10):
    assert outputs.shape[0] == labels.shape[0], "outputs and labels must have the same number of samples"
    N = outputs.shape[0]
    bins_pred, bins_y, bin_sizes = _get_bins(outputs, labels, num_bins)

    ece = 0
    oe = 0
    for conf, acc, size in zip(bins_pred, bins_y, bin_sizes):
        if size != 0:
            ece += size * abs(conf-acc)
            oe += size * (conf * max(conf - acc, 0))

    return ece /N, oe/N
    

    

def ood_uncertainty(model, device):
    """
    given the model and ood_loader, return the aleatoric uncertainty and epistemic uncertainty
    """
    svhn_loader = get_svhn_loader()
    model.eval()

    outputs, labels = get_model_outputs_and_labels(svhn_loader, model, device)

    return outputs, labels
    

    pass 




class EpochLossCounter:
    """  
    elegant way to keep track of epoch loss 
    last batch usually has less elements
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0 
        self.count = 0

    def update(self, batch_loss, n):
        self.count += n
        self.loss += batch_loss * n

    def get_epoch_loss(self):
        return self.loss / self.count


if __name__ == "__main__":
 
    #reader = make_reader("/home/slaing/ML/2nd_year/sem2/research/CIFAR10H")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_resnet_cifar(depth=20).to(device)
    model.load_state_dict(torch.load("/home/slaing/ML/2nd_year/sem2/research/models/new_net/536411_hard_False_BS_64_LR_0.04_epochs_100_depth_20_gamma_0.1_mom_0.9.pth", map_location=device))

    ood_uncertainty(model, device)

    '''
    train_loader, val_loader, test_loader = make_loaders(reader, use_hard_labels=True, batch_size=128, split_ratio=[0.8, 0.05, 0.15])
    print("loader done")
    print(evaluate_model(model, test_loader, device))
    '''