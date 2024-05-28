import torch
import torch.nn.functional as F
import math

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def pointed_kl_divergence(alpha, num_classes, target_alpha = None, beta2 = 20, device=None):
    if not device:
        device = get_device()
    ones =  torch.ones([1, num_classes], dtype=torch.float32, device=device)
    
    
    if target_alpha is not None:
        ones = target_alpha
    else:
        ones[0][0] = beta2
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def one_hot_embedding(labels, num_classes, ignore_index = None):
        
    # Convert to One Hot Encoding
    device = get_device()
    labels = labels.to(device)
    
    ignore_embedding = torch.zeros(num_classes).reshape(1,num_classes)
    y = torch.eye(num_classes)
    if ignore_index == -1:
        y = torch.cat((torch.eye(num_classes), ignore_embedding), 0)
    y = y.to(device)
    
    return y[labels]
