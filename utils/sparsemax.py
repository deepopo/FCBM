# Revised from entmax

"""
An implementation of entmax (Peters et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.

This builds on previous work with sparsemax (Martins & Astudillo, 2016).
See https://arxiv.org/pdf/1602.02068.
"""

import torch
from torch.autograd import Function

def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

def sparsemax(X, dim=-1, k=None, temperature=1.0):
    """sparsemax: normalizing sparse transform (a la softmax).

    Solves the projection:

        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    dim : int
        The dimension along which to apply sparsemax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """

    return SparsemaxFunction.apply(X, dim, k, temperature)

def _sparsemax_threshold_and_support(X, dim=-1, k=None, temperature=1.0):
    """Core computation for sparsemax: optimal threshold and support size.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.

    dim : int
        The dimension along which to apply sparsemax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - temperature
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k, temperature=temperature)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size

# class SparsemaxFunction(Function):
#     @classmethod
#     def forward(cls, ctx, X, dim=-1, k=None, temperature=1.0):
#         ctx.dim = dim
#         max_val, _ = X.max(dim=dim, keepdim=True)
#         X = X - max_val  # same numerical stability trick as softmax
#         tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k, temperature=temperature)
#         output = torch.clamp(X - tau, min=0)
#         ctx.save_for_backward(supp_size, output)
#         return output

#     @classmethod
#     def backward(cls, ctx, grad_output):
#         supp_size, output = ctx.saved_tensors
#         dim = ctx.dim
#         grad_input = grad_output.clone()
#         grad_input[output == 0] = 0

#         v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
#         v_hat = v_hat.unsqueeze(dim)
#         grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)

#         mask = (output > 0).float()
#         grad_temp = torch.sum(grad_output * mask)  # Simplified gradient
#         return grad_input, None, None, grad_temp

class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None, temperature=1.0):
        ctx.dim = dim
        ctx.temperature = temperature
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        # X_scaled = X / temperature
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k, temperature=temperature)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        temperature = ctx.temperature
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)

        grad_input = grad_input / temperature

        mask = (output > 0).float()
        active_count = supp_size.to(output.dtype)
        # 1/|S|
        term = 1.0 / active_count
        
        grad_temp_elements = torch.where(mask > 0, term, torch.zeros_like(output))
        grad_temp_elements = grad_temp_elements * grad_output
        grad_temp = torch.sum(grad_temp_elements)
        return grad_input, None, None, grad_temp