# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of RENI Losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedLoss(nn.Module):
    """
    A Loss that masks in only the top x percent of brightest pixels 
    """
    def __init__(self, topx_percent: float):
        super(MaskedLoss, self).__init__()
        self.topx_percent = topx_percent

    def forward(self, prediction, target, loss_fn=torch.nn.functional.mse_loss):

        B = target.shape[0]
        flat = target.reshape(B,-1)

        # This sould be the same for each forward and could thus be done in init
        k_rank = ( (1 - self.topx_percent) * flat.shape[1] )
        k = max(1, math.ceil(k_rank))

        threshold, _ = torch.kthvalue(flat, k, dim=1, keepdim=True)  # (B,1)
        mask = target>threshold.reshape(-1,1,1,1)

        prediction = mask*prediction
        target = mask*target
        
        return loss_fn(prediction, target)
    
class KLD3(nn.Module):
    """
    KLD Loss for isotropic sampling
    """
    def __init__(self):
        super(KLD3, self).__init__()

    def forward(self, mu, log_var): # input dims: [batch_size, latent_dim, 3], [batch_size, latent_dim, 1]
        var = torch.exp(log_var)
        kld = -0.5 * (3 + 3*log_var - mu.pow(2).sum(dim=-1, keepdim=True) - 3*var)
        kld = kld.squeeze(-1) # remove last dim (which is unary)
        # normalize wrt to latent dim and accumulate (this is what mean does.)
        kld = kld.mean(1)
        return kld

class ScaleInvariantLogLoss(nn.Module):
    def __init__(self, dim=1, reduction="mean"):
        super(ScaleInvariantLogLoss, self).__init__()
        self.dim  = dim
        if reduction == "mean":
            self.reduction_fn = torch.mean
        elif reduction == "sum":
            self.reduction_fn = torch.sum
        else:
            self.reduction_fn = lambda x: x

    def forward(self, log_predicted, log_gt, mask=None):
        R = log_predicted - log_gt
        if mask is not None:
            R = R*mask
        
        term1 = torch.mean(R**2, dim=self.dim)
        term2 = torch.mean(R, dim=self.dim)**2

        loss = term1 - term2
        loss = self.reduction_fn(loss)

        return loss

# implementation based on https://github.com/apple/ml-depth-pro/issues/60
def ssi_normalize_depth(depth):
    median = torch.median(depth)
    abs_diff = torch.abs(depth - median)  
    mean_abs_diff = torch.mean(abs_diff)
    normalized_depth = (depth - median) / mean_abs_diff
    return normalized_depth

class MultiScaleDeriLoss(nn.Module):
    def __init__(self, operator='Scharr', norm=1, scales=6, ssi=False):
        super().__init__()
        self.name = "MultiScaleDerivativeLoss"
        self.operator = operator
        self.operators = {
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[-3, 10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float).cuda(),
            },
            "Laplace": {
                'x': torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float).cuda(),
            }
        }
        self.op_x = self.operators[operator]['x']
        self.op_y = self.operators[operator]['y']
        self.ssi = ssi
        self.scales = scales

    def gradients(self, input_tensor):
        op_x, op_y = self.op_x, self.op_y
        groups = input_tensor.shape[1]
        op_x = op_x.repeat(groups, 1, 1, 1)
        op_y = op_y.repeat(groups, 1, 1, 1)
        grad_x = F.conv2d(input_tensor, op_x, groups=groups)
        grad_y = F.conv2d(input_tensor, op_y, groups=groups)
        return grad_x, grad_y

    def forward(self, prediction, target, reduction="mean"):
        loss_function = nn.L1Loss(reduction=reduction)

        if self.ssi:
            prediction_ = ssi_normalize_depth(prediction)
            target_ = ssi_normalize_depth(target)
        else:
            prediction_ = prediction
            target_ = target
        total_loss = 0.0
        for scale in range(self.scales):
            grad_prediction_x, grad_prediction_y = self.gradients(prediction_)
            grad_target_x, grad_target_y = self.gradients(target_)
            loss_x = loss_function(grad_prediction_x, grad_target_x)
            loss_y = loss_function(grad_prediction_y, grad_target_y)
            total_loss += (loss_x+loss_y)/2
            prediction_ = F.interpolate(prediction_, scale_factor=0.5)
            target_ = F.interpolate(target_, scale_factor=0.5)
        return total_loss / self.scales
