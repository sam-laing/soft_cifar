from wrappers import SNGPWrapper, DropoutWrapper, DUQWrapper
from mixup import mixup_datapoints, cutmix_datapoints

import torch
import wandb  
import argparse
import numpy as np

def train_single_epoch(model, train_loader, val_loader, test_loader, 
                       optimizer, criterion, epoch, device, args, use_mixup=False):
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
        elif args.cutmix > 0 and r < args.cutmix_prob:
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

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

