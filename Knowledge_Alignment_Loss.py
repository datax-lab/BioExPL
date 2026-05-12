import torch

def pairwise_euclidean(x, eps=1e-8):
    """
    x: (N, D)
    return: (N, N) normalized pairwise Euclidean distance matrix
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    dist2 = x_norm + x_norm.T - 2.0 * (x @ x.T)
    D = torch.sqrt(torch.clamp(dist2, min=0.0) + eps)

    # normalize for scale invariance
    return D / (D.max() + eps)

def pairwise_cosine_sim(x):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    S = x_norm @ x_norm.T
    return S

def KAL(x_sub, h, metric = 'euclidean'):
    """
    Knowledge alignment loss between input space and hidden space.

    Parameters
    ----------
    x_sub : torch.Tensor
        (B, |k|) curated features
    h : torch.Tensor
        (B, q) hidden embeddings

    Returns
    -------
    loss_KAL : torch.Tensor
        scalar knowledge alignment loss
    """
    # Pairwise distances
    if metric == 'euclidean':
        M_x = pairwise_euclidean(x_sub)
        M_h = pairwise_euclidean(h)
    # Pairwise similarity
    elif metric == 'cosine':
        M_x = pairwise_cosine_sim(x_sub)
        M_h = pairwise_cosine_sim(h)
    else:
        print('Invalid Metric')
        
    B = M_x.size(0)
    Di, Dj = torch.triu_indices(B, B, offset=1, device=M_x.device)

    diff_sq = (M_x[Di, Dj] - M_h[Di, Dj]) ** 2
    
    return diff_sq.sum()