# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import functools
import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
)
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager


@dataclass
class VENIPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VENIPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=VanillaDataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""
    eval_latent_optimisation_epochs: int = 100
    """Number of epochs to optimise latent during eval"""
    eval_latent_optimisation_lr: float = 0.1
    """Learning rate for latent optimisation during eval"""
    test_mode: Union[Literal["test", "val", "inference"], None] = None
    """overwrite test mode"""


class VENIPipeline(VanillaPipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VENIPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()  # Call grandparent class constructor ignoring parent class
        self.config = config
        self.test_mode = test_mode if self.config.test_mode is None else self.config.test_mode

        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device,
            test_mode=self.test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        if test_mode in ["val", "test"]:
            assert self.datamanager.eval_dataset is not None, "Missing validation dataset"

        num_train_data = len(self.datamanager.train_dataset)
        num_eval_data = len(self.datamanager.eval_dataset)
        self.images_per_batch = self.config.datamanager.pixel_sampler.images_per_batch

        metadata = self.datamanager.train_dataset.metadata
        if "hdr_val_images" in self.datamanager.eval_dataset.metadata:
            metadata["hdr_val_images"] = self.datamanager.eval_dataset.metadata["hdr_val_images"]

        self._model = config.model.setup(
            scene_box=None,
            num_train_data=num_train_data,
            num_eval_data=num_eval_data,
            metadata=metadata,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        self.step_of_last_latent_optimisation = 0

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """       
        # !IMPORTANT this is an invaluable tool to debug NaN in gradient
        #torch.autograd.set_detect_anomaly(True)

        ray_bundle, batch = self.datamanager.next_train(step)

        
        batch["image"] = batch["image"].to(self.model.device)
        batch["image"] = batch["image"].reshape(-1, self.model.metadata["image_height"], self.model.metadata["image_width"], 3)
        B, H, W, C = batch["image"].shape
        ray_bundle.origins = ray_bundle.origins.reshape(-1, H,W,C)
        ray_bundle.directions = ray_bundle.directions.reshape(-1, H,W,C)
        ray_bundle.camera_indices = ray_bundle.camera_indices.reshape(-1, H,W,1)

        #assert torch.unique_consecutive(ray_bundle.camera_indices).shape[0]<=B, "Camera indices should not be mixed."

        z1, mu, logvar = self.model.encode(ray_bundle, batch, step=step, return_mu_var=True)
        grad_norm = z1.norm(dim=-1).mean()
        pred1 = self._model(ray_bundle, latent_codes=z1) 
        pred1["log_var"] = logvar
        pred1["mu"] = mu
        met1 = self.model.get_metrics_dict(pred1, batch)
        loss1 = self.model.get_loss_dict(pred1, batch, met1, direction="decode", step=step)

        return pred1, loss1, {**met1, "z1_norm": grad_norm, "z1_max": z1.norm(dim=-1).max()}#, "var_mean": torch.exp(2*pred1["log_var"]).mean()} #"variational_multiplier": self.model.field.variational_multiplier}

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        #assert torch.unique(ray_bundle.camera_indices).shape[0] == 1, "only 1 image per batch is allowed!"
        
        batch["image"] = batch["image"].to(self.model.device)
        batch["image"] = batch["image"].reshape(-1, self.model.metadata["image_height"], self.model.metadata["image_width"], 3)
        B, H, W, C = batch["image"].shape
        ray_bundle.origins = ray_bundle.origins.reshape(-1, H,W,C)
        ray_bundle.directions = ray_bundle.directions.reshape(-1, H,W,C)
        ray_bundle.camera_indices = ray_bundle.camera_indices.reshape(-1, H,W,1)

        with torch.no_grad():
            z1, mu, logvar = self.model.encode(ray_bundle, batch, step=step, return_mu_var=True)
            pred1 = self._model(ray_bundle, latent_codes=z1) 
        pred1["log_var"] = logvar
        pred1["mu"] = mu
        met1 = self.model.get_metrics_dict(pred1, batch)
        loss1 = self.model.get_loss_dict(pred1, batch, met1, direction="decode", step=step)

        grad_norm = z1.norm(dim=-1).mean()
        met1 = {**met1, "z1_norm": grad_norm, "z1_max": z1.norm(dim=-1).max()}
        self.train()
        with torch.no_grad():
            return pred1, loss1, met1

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

            Args:
            step: current iteration step
        """
        self.eval()
       
        # train images
        #test_dataloader = RandIndicesEvalDataloader(
        #    input_dataset=self.datamanager.train_dataset,
        #    device=self.datamanager.device,
        #    num_workers=self.datamanager.world_size * 4,
        #)
        #camera, batch = next(test_dataloader)

        camera, batch = self.datamanager.next_eval_image(step)
        ray_bundle = camera.generate_rays(0)

        batch["image"] = batch["image"].unsqueeze(0)
        ray_bundle.origins = ray_bundle.origins.unsqueeze(0)
        ray_bundle.directions = ray_bundle.directions.unsqueeze(0)
        ray_bundle.camera_indices = ray_bundle.camera_indices.unsqueeze(0)

        with torch.no_grad():
            z1, mu, logvar = self.model.encode(ray_bundle, batch, step=step, return_mu_var=True)
            pred1 = self._model(ray_bundle, latent_codes=z1) 
        pred1["log_var"] = logvar
        pred1["mu"] = mu
        met1 = self.model.get_metrics_dict(pred1, batch)
        loss1 = self.model.get_loss_dict(pred1, batch, met1, direction="decode", step=step)

        outputs = pred1

        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch, ray_bundle)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = ray_bundle.directions.shape[-2]  # as directions is either [2, N, 3] or [N, 3]
        metrics_dict = {**metrics_dict, **loss1}

        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None, get_std: bool=False):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()

                ray_bundle = camera.generate_rays(0)

                batch["image"] = batch["image"].unsqueeze(0)
                B, H, W, C = batch["image"].shape
                ray_bundle.origins = ray_bundle.origins.unsqueeze(0)
                ray_bundle.directions = ray_bundle.directions.unsqueeze(0)
                ray_bundle.camera_indices = ray_bundle.camera_indices.unsqueeze(0)

                with torch.no_grad():
                    z1, mu, logvar = self.model.encode(ray_bundle, batch, step=step, return_mu_var=True)
                    pred1 = self._model(ray_bundle, latent_codes=z1) 
                pred1["log_var"] = logvar
                pred1["mu"] = mu
                met1 = self.model.get_metrics_dict(pred1, batch)
                loss1 = self.model.get_loss_dict(pred1, batch, met1, direction="decode", step=step)

                outputs = pred1
            
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch, ray_bundle)
                assert "num_rays" not in metrics_dict
                metrics_dict["num_rays"] = ray_bundle.directions.shape[-2]  # as directions is either [2, N, 3] or [N, 3]
                metrics_dict = {**metrics_dict, **loss1}
                
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = 8192 / (time() - inner_start)
                assert "fps" not in metrics_dict
                metrics_dict["fps"] = metrics_dict["num_rays_per_sec"] / (H * W)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([float(metrics_dict[key]) for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([float(metrics_dict[key]) for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict
