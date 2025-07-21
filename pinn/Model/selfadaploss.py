import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim import Adam, LBFGS


class SelfAdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.s_data = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # Scale factor for the loss
        self.s_pde = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.s_ic = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.s_bc = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, loss_pde, loss_ic, loss_bc):
        return (
            0.5 * torch.exp(-self.s_pde) * loss_pde +
            0.5 * torch.exp(-self.s_ic) * loss_ic +
            0.5 * torch.exp(-self.s_bc) * loss_bc
        )