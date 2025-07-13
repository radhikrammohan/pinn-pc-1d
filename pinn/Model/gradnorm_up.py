# gradnorm_loss.py

import torch
import torch.nn as nn
from typing import List

class GradNormLoss(nn.Module):
    def __init__(self, model: nn.Module, num_tasks: int, alpha: float = 1.5, apply_every: int = 100, damping: float = 0.1):
        """
        GradNormLoss constructor.

        Args:
            model (nn.Module): Shared model for gradient calculation.
            num_tasks (int): Number of tasks/losses.
            alpha (float): Balancing hyperparameter.
            apply_every (int): Frequency of GradNorm update.
            damping (float): Smoothing factor for log_alpha updates (0.0 to 1.0).
        """
        super().__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.apply_every = apply_every
        self.damping = damping
        self.log_alphas = nn.Parameter(torch.zeros(num_tasks))  # log-scale weights
        self.initial_losses = None

    def forward(self, task_losses: List[torch.Tensor], epoch: int):
        """
        Compute total weighted loss and GradNorm regularization.

        Args:
            task_losses (List[Tensor]): Scalar task losses.
            epoch (int): Current training epoch.

        Returns:
            total_loss (Tensor): Weighted task loss.
            gradnorm_loss (Tensor): GradNorm regularizer (or 0).
            alphas (Tensor): Current task weights.
        """
        alphas = torch.exp(self.log_alphas)
        weighted_losses = [a * l for a, l in zip(alphas, task_losses)]
        total_loss = sum(weighted_losses)
        gradnorm_loss = torch.tensor(0.0, device=total_loss.device)

        # Apply GradNorm only every apply_every epochs
        if epoch % self.apply_every == 0:
            shared_params = list(self.model.parameters())
            grad_norms = []

            for loss in task_losses:
                grads = torch.autograd.grad(loss, shared_params, retain_graph=True, create_graph=True, allow_unused=True)
                flat_grad = torch.cat([g.view(-1) for g in grads if g is not None])
                grad_norms.append(flat_grad.norm())

            grad_norms = torch.stack(grad_norms)
            avg_grad_norm = grad_norms.mean().detach()

            # Store initial losses at first GradNorm step
            if self.initial_losses is None:
                self.initial_losses = [loss.item() + 1e-8 for loss in task_losses]  # avoid division by 0

            loss_ratios = [task_losses[i] / self.initial_losses[i] for i in range(self.num_tasks)]
            avg_ratio = sum(loss_ratios) / len(loss_ratios)
            target_grad_norms = [avg_grad_norm * (r / avg_ratio) ** self.alpha for r in loss_ratios]
            target_grad_norms = torch.stack(target_grad_norms).detach()

            gradnorm_loss = nn.functional.l1_loss(grad_norms, target_grad_norms, reduction='sum')

            # --- Damped update to smoothen weight changes ---
            with torch.no_grad():
                # Get gradient of gradnorm loss w.r.t. log_alphas
                grad = torch.autograd.grad(gradnorm_loss, self.log_alphas, retain_graph=True)[0]
                self.log_alphas.data = self.log_alphas.data - self.damping * grad

        return total_loss, gradnorm_loss, alphas.detach()

    def get_alphas(self):
        """Return current task weights."""
        return torch.exp(self.log_alphas).detach()

    def reset(self):
        """Reset internal state (initial losses and weights)."""
        self.initial_losses = None
        with torch.no_grad():
            self.log_alphas.zero_()