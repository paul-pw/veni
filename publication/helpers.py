import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from reni.utils.helpers import load_model, device
from reni.utils.utils import find_nerfstudio_project_root
import functools
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from pathlib import Path
from reni.data.datasets.reni_dataset import RENIDataset
from veni.data.veni_pixel_sampler import VENIEquirectangularPixelSamplerConfig
from typing_extensions import Literal
import pandas as pd
import os
from datetime import datetime
from nerfstudio.engine.optimizers import Optimizers, AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager
from reni.data.dataparsers.reni_dataparser import RENIDataParserConfig
from reni.models.reni_model import RENIModel
import torch.nn.functional as f


project_root = find_nerfstudio_project_root(Path(os.getcwd()))
os.chdir(project_root)

def get_newest_model(path: str):
    path = Path(path)

    if "nerfstudio_models" in os.listdir(path):
        return path

    def parse_datetime(folder: Path):
        try:
            return datetime.strptime(folder.name, "%Y-%m-%d_%H%M%S")
        except ValueError:
            return datetime.min  # fallback if folder name doesn't match
    
    newest = max(path.glob("*/"), key=parse_datetime)

    return newest


models = [
    {"model": "outputs/ae_140/reni-ae/", "name": "Our Model 100", "model_type":"RENIAE"},
    {"model": "outputs/ae_139/reni-ae/", "name": "Our Model 49", "model_type":"RENIAE"},
    {"model": "outputs/ae_138/reni-ae/", "name": "Our Model 9", "model_type":"RENIAE"},
    

    {"model": "output/model/reni_plus_plus_models/latent_dim_100/", "name": "RENI++ 100", "model_type":"RENI"},
    {"model": "output/model/reni_plus_plus_models/latent_dim_49/", "name": "RENI++ 49", "model_type":"RENI"},
    {"model": "output/model/reni_plus_plus_models/latent_dim_9/", "name": "RENI++ 9", "model_type":"RENI"},
]

for m in models:
    m["model"] = get_newest_model(m["model"])


def load_dataloader(test_mode: Literal["val", "test"], eval_mask_path = None):
    images_per_batch = 5
    
    datamanager_config=VanillaDataManagerConfig(
        _target=VanillaDataManager[RENIDataset],
        dataparser=RENIDataParserConfig(
            data=Path("data/RENI_HDR"),
            train_subset_size=None,
            val_subset_size=None,
            convert_to_ldr=False,
            convert_to_log_domain=True,
            min_max_normalize=None, # Prior RENI implementation used (-18.0536, 11.4533) in log domain
            use_validation_as_train=False,
            augment_with_mirror=True,
            val_in_ldr=False,
            eval_mask_path = eval_mask_path,
        ),
        pixel_sampler=VENIEquirectangularPixelSamplerConfig(
            full_image_per_batch=True,
            images_per_batch=images_per_batch,  # if full_image_per_batch
            is_equirectangular=True,
        ),
        images_on_gpu=False,
        masks_on_gpu=False,
        train_num_rays_per_batch=8192,  # if not full_image_per_batch
        eval_num_rays_per_batch=8192,  # if not full_image_per_batch
    )
    
    datamanager = datamanager_config.setup(
        device=device,
        test_mode=test_mode,
        world_size=1,
        local_rank=0,
    )
    
    test_dataloader = FixedIndicesEvalDataloader(
        input_dataset=datamanager.eval_dataset,
        device=device,
        num_workers=4,
    )
    num_train_data = len(datamanager.train_dataset)
    num_eval_data = len(datamanager.eval_dataset)

    metadata = datamanager.train_dataset.metadata
    
    return test_dataloader, metadata, num_train_data, num_eval_data

def predict(camera, model, batch=None, latents=None):
    assert (batch==None)!=(latents==None), "Either batch of latents have to be defined!"

    ray_bundle = camera.generate_rays(0)
    ray_bundle.origins = ray_bundle.origins.unsqueeze(0)
    ray_bundle.directions = ray_bundle.directions.unsqueeze(0)
    ray_bundle.camera_indices = ray_bundle.camera_indices.unsqueeze(0)

    all_cameras = ray_bundle.camera_indices.flatten(end_dim=1)
    unique_cameras, inverse = torch.unique(all_cameras, return_inverse=True)

    if isinstance(model, RENIModel):
        if batch is not None:
            # Cannot do this for RENI++ model.
            return None, None, None, None
        B,H,W,C = ray_bundle.directions.shape
        ray_bundle.origins = ray_bundle.origins.reshape(-1, 3)
        ray_bundle.directions = ray_bundle.directions.reshape(-1, 3)
        ray_bundle.camera_indices = ray_bundle.camera_indices.reshape(-1)

        latents = latents.repeat_interleave(H*W, dim=0)

    if batch is not None:
        batch_unsqueeze = {"image": batch["image"].unsqueeze(0)}
        latents = model.encode(ray_bundle, batch_unsqueeze).detach()

    with torch.no_grad():    
        pred1 = model(ray_bundle, latent_codes=latents)
        if batch is not None: 
            if not isinstance(model, RENIModel):
                metrics_dict, images_dict = model.get_image_metrics_and_images(pred1, batch, ray_bundle)
            else: 
                metrics_dict, images_dict = model.get_image_metrics_and_images(pred1, batch)
        else: 
            images_dict = None
            metrics_dict = None
        
    return latents, pred1, metrics_dict, images_dict


def rotate_image(image, angle, get_angle = False):
    img_width = image.shape[1]
    angle_rad = np.deg2rad(angle)
    num_cols_to_roll = int(np.round(img_width * angle_rad / (2 * np.pi)))
    rotated_angle = float(num_cols_to_roll)/float(img_width)*360.0
    image = torch.roll(image, -num_cols_to_roll, dims=1)
    if not get_angle:
        return image
    return image, rotated_angle
    

def optimize_latents_of_image(image_to_solve, inital_latent, model, camera, lr=(1e-1, 1e-2), steps=500, mask = None):
    ray_bundle = camera.generate_rays(0)
    ray_bundle.origins = ray_bundle.origins.unsqueeze(0)
    ray_bundle.directions = ray_bundle.directions.unsqueeze(0)
    ray_bundle.camera_indices = ray_bundle.camera_indices.unsqueeze(0)
    H, W = model.metadata["image_height"], model.metadata["image_width"]
    
    latent_codes = torch.nn.Parameter(inital_latent.clone().detach())
    if isinstance(model, RENIModel):
        ray_bundle.origins = ray_bundle.origins.reshape(-1,3)
        ray_bundle.directions = ray_bundle.directions.reshape(-1,3)
        ray_bundle.camera_indices = ray_bundle.camera_indices.reshape(-1)

    if mask is not None: 
        mask = mask.to(model.device)
    else:
        mask = torch.ones_like(image_to_solve, device=model.device)
    image_to_solve = image_to_solve.to(model.device)

    # setup optimiser
    optimiser_config = {
        "latents": {
            "optimizer": AdamOptimizerConfig(lr=lr[0], eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=lr[1], max_steps=steps),
        },
    }
    
    param_group = {"latents": [latent_codes]}
    optimizer = Optimizers(optimiser_config, param_group)
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("[blue]Loss: {task.fields[loss]}"),
        TextColumn("[green]LR: {task.fields[lr]}")
    ) as progress:
        task = progress.add_task("[green]Fitting latents... ", total=steps, loss="", lr="", latent_error="")
    
        for step in range(steps):
            if isinstance(model, RENIModel):
                model_outputs = model(ray_bundle, latent_codes=latent_codes.repeat_interleave(H*W, dim=0))
            else:
                model_outputs = model(ray_bundle, latent_codes=latent_codes)
            model_outputs["rgb"] = model_outputs["rgb"].reshape(-1,H,W,3)
            
            
            #loss_dict = model.get_loss_dict({"rgb": model_outputs["rgb"]*mask.unsqueeze(0), "log_var": torch.tensor([0.0]), "mu": torch.tensor([0.0])}, {"image": image_to_solve.unsqueeze(0)*mask.unsqueeze(0)}, None)
            #loss = functools.reduce(torch.add, loss_dict.values())

            loss = f.mse_loss(model_outputs["rgb"]*mask.unsqueeze(0), image_to_solve.unsqueeze(0)*mask.unsqueeze(0))
    
            optimizer.zero_grad_all()
            loss.backward()
            optimizer.optimizer_step("latents")
            optimizer.scheduler_step("latents")
            
            progress.update(
                task,
                advance=1,
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.schedulers['latents'].get_last_lr()[0]:.8f}"
            )
    return latent_codes

def lerp(start, end, t):
    return start * (1 - t) + end * t
