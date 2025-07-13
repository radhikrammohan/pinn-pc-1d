"""
GradNormLoss: A PyTorch implementation of GradNorm for multi-loss balancing
Adapted for Physics-Informed Neural Networks (PINNs) and general multi-task learning

Reference:
Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (ICML 2018)
https://arxiv.org/abs/1711.02257
"""

import torch
import torch.nn as nn
from typing import List

class GradNormLoss(nn.Module):
    def __init__(self, model: nn.Module, num_tasks: int, alpha: float = 1.5):
        """
        GradNormLoss constructor.

        Args:
            model (nn.Module): The shared neural network whose parameters are used to compute gradients.
            num_tasks (int): Number of individual loss components (e.g., data, PDE, BC).
            alpha (float): GradNorm balancing strength (default: 1.5).
        """
        super().__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.log_alphas = nn.Parameter(torch.zeros(num_tasks))  # log-scaled learnable weights
        self.initial_losses = None  # to store initial loss values for normalization

    def forward(self, task_losses: List[torch.Tensor], epoch: int):
        """
        Forward pass to compute total weighted loss and GradNorm loss.

        Args:
            task_losses (List[Tensor]): List of scalar task losses [L1, L2, ..., Ln].
            epoch (int): Current training epoch (used to store initial losses).

        Returns:
            total_loss (Tensor): Weighted sum of task losses.
            gradnorm_loss (Tensor): Loss used to train log_alphas.
            alphas (Tensor): Current weights for each loss (detached from graph).
        """
        alphas = torch.exp(self.log_alphas)
        weighted_losses = [a * l for a, l in zip(alphas, task_losses)]
        total_loss = sum(weighted_losses)

        shared_params = list(self.model.parameters())
        grad_norms = []

        for loss_i in task_losses:
            grads = torch.autograd.grad(loss_i, shared_params, retain_graph=True, create_graph=True, allow_unused=True)
            flat_grad = torch.cat([g.view(-1) for g in grads if g is not None])
            grad_norms.append(flat_grad.norm())

        grad_norms = torch.stack(grad_norms)
        avg_grad_norm = grad_norms.mean().detach()

        if epoch == 0:
            self.initial_losses = [l.item() for l in task_losses]

        loss_ratios = [task_losses[i] / self.initial_losses[i] for i in range(self.num_tasks)]
        avg_ratio = sum(loss_ratios) / len(loss_ratios)
        target_grad_norms = [avg_grad_norm * (r / avg_ratio) ** self.alpha for r in loss_ratios]
        target_grad_norms = torch.stack(target_grad_norms).detach()

        gradnorm_loss = nn.functional.l1_loss(grad_norms, target_grad_norms, reduction='sum')

        return total_loss, gradnorm_loss, alphas.detach()

    def get_alphas(self):
        """Returns the current loss weights (alpha_i = exp(log_alpha_i))."""
        return torch.exp(self.log_alphas).detach()

    def reset(self):
        """Resets the internal state (useful for reinitializing initial losses)."""
        self.initial_losses = None
        with torch.no_grad():
            self.log_alphas.zero_()