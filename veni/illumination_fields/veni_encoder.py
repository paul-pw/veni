
from typing import Literal, Type, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import contextlib

import torch
from torch import nn, Tensor
from jaxtyping import Float
from einops.layers.torch import Rearrange

from nerfstudio.configs.base_config import InstantiateConfig

from reni.field_components.siren import Siren
from reni.field_components.film_siren import FiLMSiren
from reni.field_components.activations import ExpLayer
from reni.field_components.transformer_decoder import Decoder
from reni.field_components.field_heads import RENIFieldHeadNames

from veni.field_components.vn_layers import VNInvariant, VNLinear, VNReLU, VNTransformerEncoder
from veni.field_components.sox_layers import SOxTransformerEncoder, SOxLinear, SOxReLU

@dataclass
class VENIEncoderConfig(InstantiateConfig):
    """configuration for Encoder"""

    _target: Type = field(default_factory=lambda: VENIEncoder)
    """target class to instanciate"""
    pooling: Literal["cls", "mean"] = "cls"
    """use class token or mean pooling"""
    learnable_cls: bool = True
    """is class token learnable or not (only invariant axes are learnable)"""
    invariant_axes: set[int] = (2,)
    """which coordinate axes are rotation invariant"""
    reduce_feature_to_coord: Literal["early_learned", "late_split", "late_learned"] = "late_split"
    """if and when to reduce feature dim to coord dim"""
    transformer_type: Literal["VN", "SOx"] = "VN"
    """which transformer Type to use"""
    inner_dim: int = 80
    dim_head: int = 16
    heads: int = 5
    depth: int = 6
    project_in_type: Literal["VN", "SOx"] = "VN"
    project_out_type: Literal["VN", "SOx"] = "SOx"
    variational: bool = False
    mirror: Literal["none", "early", "late"] = "none"
    """without mirroring the latent codes rotate counter to the features (when the features are rotated.), so we need mirror along the x or y axis to reverse the rotation of the latent codes, so features and latent codes rotate the same way"""
    
class VENIEncoder(nn.Module):

    def __init__(
        self,
        config: VENIEncoderConfig,
        patch_size,
        latent_dim
    ) -> None:
        super().__init__()
        
        self.config = config
        self.pool = self.config.pooling # cls
        self.learnable_cls = self.config.learnable_cls # True
        self.dim_feat = 3
        self.dim_coor = 3
        self.dim_coor_total = 6
        dim_head=self.config.dim_head # 16
        heads=self.config.heads # 5
        depth=self.config.depth # 6
        transformer_dim = self.config.inner_dim # 80
        self.reduce_feature_to_coord = self.config.reduce_feature_to_coord
        transformer_type = self.config.transformer_type
        self.variational = self.config.variational

        invariant_axes_coord = list(self.config.invariant_axes)
        equivariant_axes = [x for x in range(3) if x not in invariant_axes_coord] 
        self.mirror_axis = equivariant_axes[0]
        # make feature axes always be rotation invariant (feats are alwys the last axes.)
        invariant_axes_feats = list(range(self.dim_coor, self.dim_coor+self.dim_feat))
        if self.reduce_feature_to_coord == "early_learned":
            self.inner_invariant_axes = invariant_axes_coord
            dim_coor_inner = self.dim_coor
        else:
            self.inner_invariant_axes = invariant_axes_coord + invariant_axes_feats
            dim_coor_inner = self.dim_coor_total

        assert not (self.config.project_in_type == "VN" and self.reduce_feature_to_coord =="early_learned"), "VN can not reduce feature dim to coord, so this is not a valid config, use project_in_type=SOx"
        assert not (self.config.project_out_type == "VN" and self.reduce_feature_to_coord =="late_learned"), "VN can not reduce feature dim to coord, so this is not a valid config, use project_out_type=SOx or reduce with late_split"
        
        if self.config.project_in_type == "VN":
            self.vn_proj_in = nn.Sequential(
                nn.Identity() if patch_size else Rearrange('... c -> ... 1 c'),
                VNLinear(patch_size or 1, transformer_dim)
            )
        else:
            self.vn_proj_in = nn.Sequential(
                nn.Identity() if patch_size else Rearrange('... c -> ... 1 c'),
                SOxLinear(patch_size or 1,
                          transformer_dim, 
                          equivariant_axes,
                          invariant_axes_coord + invariant_axes_feats, 
                          invariant_axes_out=self.inner_invariant_axes),
            )

        if self.config.project_out_type == "VN":
            create_project_out = lambda: VNLinear(transformer_dim,latent_dim)
        else: 
            if self.reduce_feature_to_coord == "late_split":
                invariant_axes_out = self.inner_invariant_axes
            else:
                invariant_axes_out = invariant_axes_coord
            create_project_out = lambda: SOxLinear(transformer_dim,latent_dim,equivariant_axes, self.inner_invariant_axes, invariant_axes_out=invariant_axes_out)

        self.project_out = create_project_out()
        if self.variational:
            self.project_out_logvar = create_project_out()
            self.reduce_out_logvar = lambda x: torch.linalg.vector_norm(x[...,equivariant_axes], dim=-1, keepdim=True)+x[...,invariant_axes_coord].sum(dim=-1, keepdim=True)
        

        if self.pool == "cls" and self.learnable_cls:
            # SOx equivariant/invariant class token
            self.cls_inv = nn.Parameter(torch.zeros(1,1,transformer_dim, len(self.inner_invariant_axes)))
        
        if transformer_type == "VN":
            self.encoder = VNTransformerEncoder(
                dim = transformer_dim,
                depth = depth,
                dim_head = dim_head,
                heads = heads,
                dim_coor = dim_coor_inner,
            )
        else:
            self.encoder = SOxTransformerEncoder(
                dim = transformer_dim,
                equivariant_axes = equivariant_axes, 
                invariant_axes = self.inner_invariant_axes,
                depth = depth,
                dim_head = dim_head,
                heads = heads,
                dim_coor = dim_coor_inner,
            )

        print("ENCODER_PARAMETERS",sum(p.numel() for p in self.parameters()))


    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar*0.5)
            eps = torch.normal(torch.zeros(mu.shape, device=mu.device), 1) # normal distribution with mean=0 and std=1
            return eps*std+mu 
        else:
            return mu

    def forward(self, directions, colors,mask=None) -> float: 
        """
        """

        x = torch.cat((directions, colors), dim=-1) # Shape: [Batch Size, Squence Length, 3+dim_feats]

        B = x.shape[0]
        C = x.shape[-1]  
        assert C == self.dim_coor_total, "Last dim must match configured input dim"

        if self.config.mirror == "early":
            # Mirror along the selected mirror axis
            x[...,self.mirror_axis] = -x[...,self.mirror_axis]

        x = self.vn_proj_in(x) # Shape: [Batch Size, Sequence Lenght, Dim, 3+dim_feats]
        
        if self.pool=="cls":
            self.cls_token = torch.zeros(1,1,x.shape[-2],x.shape[-1], device=x.device)
            if self.learnable_cls:
                self.cls_token[...,self.inner_invariant_axes] = self.cls_inv
            cls_token = self.cls_token.expand(B,-1,-1,-1)
            x = torch.cat((cls_token, x), dim=1) # Shape: [Batch Size, Sequence Length + 1, Dim, 3+dim_feats]


        # The actual Transformer Encoder.
        x = self.encoder(x, mask=mask)

        if self.pool == "cls":
            x = x[:,0] # Shape: [Batch Size, Dim, 3+dim_feats]
        elif self.pool == "mean":
            x = x.mean(dim=1) # Shape: [Batch Size, Dim, 3+dim_feats]
        else: 
            raise NotImplementedError()


        mu = self.project_out(x)
        mu = mu[...,:self.dim_coor] # this only takes account when using late split, with other configurations, the last dim is already maximal dim_coor

        logvar = None
        if self.variational:
            logvar = self.project_out_logvar(x)
            logvar = logvar[...,:self.dim_coor]
            logvar = self.reduce_out_logvar(logvar)
            latent = self.reparametrize(mu, logvar)
        else:
            latent = mu
    
        if self.config.mirror == "late":
            # Mirror along the selected mirror axis
            latent[...,self.mirror_axis] = -latent[...,self.mirror_axis]
            mu[...,self.mirror_axis] = -mu[...,self.mirror_axis]


        return latent, mu, logvar



