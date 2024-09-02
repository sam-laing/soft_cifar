import numpy as np
import torch

def mixup_datapoints(x, y, device, alpha=0.2):
    '''
    given a minibatch of data x and labels y, 
    generate an idx of samples to mixup with

    p is the amount of weight to put on the first sample
    it will be generated from a beta distribution with parameter alpha

    '''
    idx = torch.randperm(x.shape[0]).to(device)
    lam = np.random.beta(alpha, alpha)

    x = lam * x + (1 - lam) * x[idx]
    y = lam * y + (1 - lam) * y[idx]
    return x, y

def cutmix_datapoints(x, y, device, alpha=0.2):
    """
    given minibatch x,y
    do a randomized cutmix operation on x
    and take convex combination on y with area of the cut region
    """
    rand_idx = torch.randperm(x.shape[0]).to(device)
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # fill in the cut region with the data from the random index
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_idx, :, bbx1:bbx2, bby1:bby2]
    lam1 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y = lam1 * y + (1 - lam1) * y[rand_idx]

    return x, y


def rand_bbox(size, lam):
    W, H = size[-2], size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2