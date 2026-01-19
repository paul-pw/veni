"""
Implementation of VENI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Type, Literal, Optional, Union
import functools
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn

import torch
import torch.nn as nn
import torch.nn.functional as fnn
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB

from veni.model_components.losses import KLD3, ScaleInvariantLogLoss, MultiScaleDeriLoss, MaskedLoss
from veni.model_components.loss_schedulers import  LossWeightScheduler, LossWeightSchedulerConfig
from veni.model_components.loss_schedulers import  LossWeightScheduler, LossWeightSchedulerConfig
from veni.model_components.samplers import equirectangular_sampling, uniform_directions, uniform_sunflower_patches, sunflower_3d, double_sunflower, pitch_yaw_to_rot, line_sampling
from veni.illumination_fields.veni_encoder import VENIEncoderConfig

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.optimizers import OptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import SchedulerConfig
import math


CONSOLE = Console(width=120)

@dataclass
class VENIModelConfig(ModelConfig):
    """Vanilla Model Config"""
    _target: Type = field(default_factory=lambda: VENIModel)
    """loss scheduler config"""
    loss_inclusions: Dict[str, Literal[True, False]] = to_immutable_dict(
        {
            "log_mse_loss": False,
            "hdr_mse_loss": False,
            "ldr_mse_loss": False,
            "kld_loss": False,
            "cosine_similarity_loss": False,
            "scale_inv_loss": False,
            "multi_scale_derivative_loss": False,
        }
    )
    """Which losses to include in the training"""
    loss_coefficients: Dict[str, LossWeightSchedulerConfig] = to_immutable_dict(
        {
            "log_mse_loss": field(default_factory=lambda: LossWeightSchedulerConfig),
            "hdr_mse_loss": field(default_factory=lambda: LossWeightSchedulerConfig),
            "ldr_mse_loss": field(default_factory=lambda: LossWeightSchedulerConfig),
            "kld_loss": field(default_factory=lambda: LossWeightSchedulerConfig),
            "cosine_similarity_loss": field(default_factory=lambda: LossWeightSchedulerConfig),
            "scale_inv_loss": field(default_factory=lambda: LossWeightSchedulerConfig),
            "multi_scale_derivative_loss": field(default_factory=lambda: LossWeightSchedulerConfig),
        }
    )
    encoder: VENIEncoderConfig = field(default_factory=lambda: VENIEncoderConfig)
    patch_size: int|None = 256
    sampling: str = "double_sunflower"
    samples_per_img: int = 64
    fov: int = 33
    """loss coefficients with LossWeightScheduling"""
    # be careful this field overrides the imported field!
    field: SphericalFieldConfig = field(default_factory=lambda: SphericalFieldConfig)
    """Field configuration"""
        
class VENIModel(Model):
    """Rotation-Equivariant Neural Illumination Model

    Args:
        config: Model config
    """

    config: VENIModelConfig

    def __init__(
        self,
        config: VENIModelConfig,
        **kwargs,
    ) -> None:
        self.metadata = kwargs["metadata"]
        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        normalisations = {"min_max": self.metadata["min_max"], "log_domain": self.metadata["convert_to_log_domain"]}
        self.field = self.config.field.setup(normalisations=normalisations)
        self.encoder = self.config.encoder.setup(patch_size=self.config.patch_size, latent_dim=self.field.latent_dim)
        
        self.loss_coefficients = {}
        for k,v in self.config.loss_coefficients.items():
            self.loss_coefficients[k] = v.setup()

        # losses 
        if self.config.loss_inclusions["kld_loss"] != False:
            self.kld_loss = KLD3()
        if self.config.loss_inclusions["cosine_similarity_loss"] != False:
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        if self.config.loss_inclusions["scale_inv_loss"] != False:
            self.scale_invariant_loss = ScaleInvariantLogLoss(dim=(1,2), reduction=None) # dim is width and height dims.
        if self.config.loss_inclusions["multi_scale_derivative_loss"] != False:
            self.mage_loss = MultiScaleDeriLoss(scales=2) 
        
        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def forward(self, ray_bundle: RayBundle, latent_codes: torch.Tensor, rotation: Optional[torch.Tensor] = None, direction: Literal["encode", "decode"] = "decode") -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        return self.get_outputs(ray_bundle,latent_codes, rotation, direction=direction)

    def create_ray_samples(self, origins, directions, camera_indices) -> RaySamples:
        """Create ray samples from a ray bundle"""

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=torch.zeros_like(camera_indices),
                ends=torch.ones_like(camera_indices),
                pixel_area=torch.ones_like(camera_indices),
            ),
            camera_indices=camera_indices,
        )

        return ray_samples

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["field"] = list(self.field.parameters())
        param_groups["encoder"] = list(self.encoder.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle,  latent_codes, rotation: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None, direction: Literal["encode", "decode"] = "decode"):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        ray_samples = self.create_ray_samples(ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)

        field_outputs = self.field.forward(ray_samples=ray_samples, latent_codes=latent_codes,rotation=rotation, scale=scale, direction=direction)

        outputs = {
            "rgb": field_outputs[RENIFieldHeadNames.RGB],
        }

        if RENIFieldHeadNames.MU in field_outputs:
            outputs["mu"] = field_outputs[RENIFieldHeadNames.MU]
        if RENIFieldHeadNames.LOG_VAR in field_outputs:
            outputs["log_var"] = field_outputs[RENIFieldHeadNames.LOG_VAR]

        return outputs

    def encode(self, ray_bundle, batch, step=0, return_mu_var=False, **kwargs):
        image = batch["image"].to(self.device)
        B, H, W, C = image.shape
       
        sampling = self.config.sampling 
        sample_count = self.config.samples_per_img
        patch_size = self.config.patch_size
        fov = self.config.fov

        if sampling == "regular_equirectangular":
            ray_samples = self.create_ray_samples(ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)
            directions = ray_samples.frustum.directions[:,::3,::3,:]
            colors = image[:,::3,::3,:]
            directions = directions.reshape(B, -1, 3)
            colors = colors.reshape(B,-1,3)
        elif sampling == "uniform_random":
            num_rays = sample_count # 1024
            directions = uniform_directions((B,num_rays,3), device=self.device)
            colors = equirectangular_sampling(directions, image)
        elif sampling == "random_sunflower_patches":
            num_patches = sample_count # 512
            rays_per_sunflower = patch_size # 25
            fov = fov # 17
            directions = uniform_sunflower_patches(B, num_patches, rays_per_sunflower, fov, device=self.device)
            colors = equirectangular_sampling(directions.reshape(B,-1,3), image)
            colors = colors.reshape(directions.shape)
        elif sampling == "sunflower_directions":
            num_rays = sample_count # 1024
            directions = sunflower_3d(num_rays, device=self.device).expand(B,-1,-1)
            colors = equirectangular_sampling(directions, image)
            colors = colors.reshape(directions.shape)
        elif sampling == "double_sunflower":
            num_patches = sample_count # 64
            rays_per_sunflower = patch_size # 256
            fov = fov
            directions = double_sunflower(B, num_patches, rays_per_sunflower, fov, device=self.device)
            colors = equirectangular_sampling(directions.reshape(B,-1,3), image)
            colors = colors.reshape(directions.shape)
        elif sampling == "double_sunflower_translation":
            num_patches = sample_count # 64
            rays_per_sunflower = patch_size # 256
            fov = fov
            directions = double_sunflower(B, num_patches, rays_per_sunflower, fov, device=self.device)
            # To get  better translation equivariance, we do rotate the directions. (_translation augmentation_)
            # Because Patches have up direction, we cannot have pitch rotation not be too high, so up is somewhat preserved.
            if self.training:
                R = pitch_yaw_to_rot(torch.randn(B, device=self.device)*(fov/360.0)*math.pi, torch.rand(B, device=self.device)*2*math.pi, pitch_axis=0, yaw_axis=2)
                R = R.unsqueeze(1).unsqueeze(1)
                directions = torch.einsum('...ij,...j->...i', R, directions) 

            colors = equirectangular_sampling(directions.reshape(B,-1,3), image)
            colors = colors.reshape(directions.shape)
        elif sampling == "line_sampling":
            directions = line_sampling(B, sample_count, patch_size, device=self.device)
            if self.training:
                R = pitch_yaw_to_rot(torch.zeros(B, device=self.device), torch.rand(B, device=self.device)*2*math.pi, pitch_axis=0, yaw_axis=2)
                R = R.unsqueeze(1).unsqueeze(1)
                directions = torch.einsum('...ij,...j->...i', R, directions) 

            colors = equirectangular_sampling(directions.reshape(B,-1,3), image)
            colors = colors.reshape(directions.shape)
        else: 
            raise NotImplementedError()
 
        z, mu, logvar = self.encoder(directions, colors)
        if return_mu_var:
            return z, mu, logvar
        return z

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        device = outputs["rgb"].device
        gt_image = batch["image"].to(device)
        pred_image = outputs["rgb"]

        if self.config.loss_inclusions["scale_inv_loss"]:
            # estimate scale using least squares
            scale = (gt_image * pred_image).sum() / (pred_image * pred_image).sum()
            pred_image = scale * pred_image

        gt_image = self.field.unnormalise(gt_image)
        pred_image = self.field.unnormalise(pred_image)

        psnr = self.psnr(preds=pred_image, target=gt_image)

        metrics_dict = {"psnr": psnr}
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None, reduction="mean", direction: Literal["encode", "decode"]="decode", step=0) -> Dict[str, torch.Tensor]: # here, the loss is computed
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device

        batch["image"] = batch["image"].to(device)

        loss_dict = {}
        if reduction == "mean":
            reduction_fn = torch.mean
        elif reduction == "sum":
            reduction_fn = torch.sum
        else:
            raise ValueError("Reduction must be one of 'mean' or 'sum'")

        B,H,W,C = outputs["rgb"].shape
        # add some small value so values at the edges also get trained
        mask = (torch.sin(torch.linspace(0,math.pi,H,device=device)).reshape(H,1,1)+0.001).clamp(max=1.0)

        # Unlike original RENI implementation, the sineweighting
        # is implemented by the ray sampling so no need to modify losses
        if self.config.loss_inclusions["log_mse_loss"] == True:
            log_mse_loss = fnn.mse_loss(outputs["rgb"]*mask, batch["image"]*mask, reduction=reduction)
            loss_dict["log_mse_loss"] = log_mse_loss

        if self.config.loss_inclusions["hdr_mse_loss"] == True:
            hdr_mse_loss = fnn.mse_loss(torch.exp(outputs["rgb"])*mask, torch.exp(batch["image"])*mask, reduction=reduction)
            loss_dict["hdr_mse_loss"] = hdr_mse_loss

        if self.config.loss_inclusions["ldr_mse_loss"] == True:
            ldr_mse_loss = fnn.mse_loss(outputs["rgb"]*mask, batch["image"]*mask, reduction=reduction)
            loss_dict["ldr_mse_loss"] = ldr_mse_loss

        if self.config.loss_inclusions["kld_loss"] == True and self.training:
            kld_loss = self.kld_loss(outputs["mu"], outputs["log_var"])
            loss_dict["kld_loss"] = reduction_fn(kld_loss)

        if self.config.loss_inclusions["cosine_similarity_loss"] == True:
            similarity = self.cosine_similarity(outputs["rgb"], batch["image"])
            similarity = similarity * mask.squeeze(-1)

            cosine_similarity_loss = reduction_fn(1.0 - similarity)
            loss_dict["cosine_similarity_loss"] = cosine_similarity_loss

        if self.config.loss_inclusions["scale_inv_loss"] == True:
            scale_inv_loss = self.scale_invariant_loss(outputs["rgb"], batch["image"],mask=mask)
            loss_dict["scale_inv_loss"] = reduction_fn(scale_inv_loss)
        
        if self.config.loss_inclusions["multi_scale_derivative_loss"] == True:
            pred = torch.swapdims(outputs["rgb"]*mask, 1, -1) # channels are expected to be at dim 1
            target = torch.swapdims(batch["image"]*mask, 1, -1)
            mage_loss = self.mage_loss(pred, target, reduction=reduction)
            loss_dict["multi_scale_derivative_loss"] = mage_loss
        
        loss_coefficients_at_step = {k: v.weight(step) for k,v in self.loss_coefficients.items()}
        # scale with loss_coefficients
        loss_dict = misc.scale_dict(loss_dict, loss_coefficients_at_step)
        return loss_dict

    @torch.no_grad()
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ray_bundle, estimate_scale = True
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        device = outputs["rgb"].device
        batch["image"] = batch["image"].to(device)

        if self.metadata["val_in_ldr"]:
            gt_image = self.metadata["hdr_val_images"][batch["indices"][0, 0]].reshape(-1, 3)  # [num_rays, 3]
            gt_image = gt_image.to(device)
        else:
            gt_image = batch["image"]  # [num_rays, 3]

        pred_image = outputs["rgb"]  # [num_rays, 3]

        # reshape to [H, W, 3]
        gt_image = gt_image.reshape(self.metadata["image_height"], self.metadata["image_width"], 3)
        pred_image = pred_image.reshape(self.metadata["image_height"], self.metadata["image_width"], 3)

        if self.config.loss_inclusions["scale_inv_loss"] in [True, "True"] or estimate_scale:
            # estimate scale using least squares
            scale = (gt_image * pred_image).sum() / (pred_image * pred_image).sum()
            pred_image = scale * pred_image

        gt_image = self.field.unnormalise(gt_image)
        pred_image = self.field.unnormalise(pred_image)

        # converting to grayscale by taking the mean across the color dimension
        gt_image_gray = torch.mean(gt_image, dim=-1)
        pred_image_gray = torch.mean(pred_image, dim=-1)

        # reshape to H, W
        gt_image_gray = gt_image_gray.reshape(self.metadata["image_height"], self.metadata["image_width"], 1)
        pred_image_gray = pred_image_gray.reshape(self.metadata["image_height"], self.metadata["image_width"], 1)

        gt_min, gt_max = torch.min(gt_image_gray).item(), torch.max(gt_image_gray).item()

        combined_log_heatmap = torch.cat([gt_image_gray, pred_image_gray], dim=1)

        combined_log_heatmap = colormaps.apply_depth_colormap(
            combined_log_heatmap,
            near_plane=gt_min,
            far_plane=gt_max,
        )

        # create difference image
        difference = torch.abs(gt_image - pred_image)

        # i.e. we are not already in LDR space
        if not self.metadata["convert_to_ldr"]:
            # convert from linear HDR to sRGB for viewing
            gt_image_ldr = linear_to_sRGB(gt_image, use_quantile=True)
            pred_image_ldr = linear_to_sRGB(pred_image, use_quantile=True)
        else:
            gt_image_ldr = gt_image
            pred_image_ldr = pred_image

        if "mask" in batch:
            mask = batch["mask"].reshape(self.metadata["image_height"], self.metadata["image_width"], 1).expand_as(
                gt_image_ldr
            ).to(device) # [H, W, 3]
            # we should mask gt_image_ldr to show only the pixels that were used in the loss
            masked_gt_image_ldr = gt_image_ldr * mask
            combined_rgb = torch.cat([gt_image_ldr, masked_gt_image_ldr, pred_image_ldr], dim=1)
        else:
            combined_rgb = torch.cat([gt_image_ldr, pred_image_ldr], dim=1)

        random_image = self(ray_bundle, latent_codes=torch.randn(1,self.field.latent_dim,3,device=device)) 
        random_image = linear_to_sRGB(self.field.unnormalise(random_image["rgb"].squeeze(0)), use_quantile=True)
        
        images_dict = {}

        images_dict["img"] = combined_rgb
        images_dict["heatmap"] = combined_log_heatmap
        images_dict["difference"] = difference
        images_dict["random_sample"] = random_image

        # COMPUTE METRICS
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_image = gt_image.unsqueeze(0).permute(0, 3, 1, 2)
        pred_image = pred_image.unsqueeze(0).permute(0, 3, 1, 2)
        gt_image_ldr = gt_image_ldr.unsqueeze(0).permute(0, 3, 1, 2)
        pred_image_ldr = pred_image_ldr.unsqueeze(0).permute(0, 3, 1, 2)

        metrics_dict = {}

        metrics_dict["psnr_hdr"] = self.psnr(preds=pred_image, target=gt_image)
        metrics_dict["ssim_hdr"] = self.ssim(preds=pred_image, target=gt_image)

        # for lpips we need to convert to 0 to 1 using image.min() and image.max()
        gt_image = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min())
        pred_image = (pred_image - pred_image.min()) / (pred_image.max() - pred_image.min())
        metrics_dict["lpips_hdr"] = self.lpips(pred_image, gt_image)

        # if we are not already learning in LDR space
        if not self.metadata["convert_to_ldr"]:
            metrics_dict["psnr_ldr"] = self.psnr(preds=pred_image_ldr, target=gt_image_ldr)
            metrics_dict["ssim_ldr"] = self.ssim(preds=pred_image_ldr, target=gt_image_ldr)
            metrics_dict["lpips_ldr"] = self.lpips(pred_image_ldr, gt_image_ldr)

        return metrics_dict, images_dict
