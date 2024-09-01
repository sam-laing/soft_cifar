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


### cutmix
def create_bbox(image_size, lam):
    '''
    create a bounding box for cutmix
    '''
    W, H = image_size
    
    pass
    
