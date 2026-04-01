import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import re
from typing import Optional
from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import colormaps, misc

from veni.configs.veni_gon_config import VENIGon
from veni.configs.veni_config import VENI
from reni.configs.reni_config import RENIField
from reni.configs.sh_sg_envmap_configs import SHField, SGField
from veni.pipelines.veni_gon_pipeline import VENIGONPipeline
#from reni.field_components.field_heads import RENIFieldHeadNames
from reni.data.datamanagers.reni_datamanager import RENIDataManager
from reni.data.reni_pixel_sampler import RENIEquirectangularPixelSamplerConfig,RENIEquirectangularPixelSampler
from reni.utils.utils import find_nerfstudio_project_root, rot_z, rot_y
#from reni.utils.colourspace import linear_to_sRGB
#import functools
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from typing import Literal
#project_root = find_nerfstudio_project_root(Path(os.getcwd()))
#os.chdir(project_root)
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from veni.model_components.loss_schedulers import ConstantLossWeightConfig, LinearLossWeightSchedulerConfig


# setup config
world_size = 1
local_rank = 0
device = 'cuda:0'

#def set_attr_from_config(obj, config):
#    
#    is_dict_config = any([hasattr(obj, k) for k,v in config.items()])
#    
#    if not isinstance(config, dict) or:
#        setattr(obj, config)
#        return
#        
#    for k,v in config.items():
#        set_attr_from_config(getattr(obj, k), v)


def load_model(
      load_dir: Path, 
      load_step: Optional[int] = None, 
      datapath = Path("../data/RENI_HDR").absolute(), 
      custom_division_factor=None, 
      variational=False, 
      model_type: Literal['RENIGon', 'RENI', 'VENI']="RENIGon",
      model_only = False,
      metadata={},
      num_train_data=None, 
      num_eval_data=None,
      load_state=True):
    ckpt_dir = load_dir / 'nerfstudio_models'
    def clean_and_load_yaml(yaml_content):
        # Remove !!python related tags
        cleaned_content = re.sub(r'!!python[^\s]*', '', yaml_content)
        
        # Load the cleaned content
        return yaml.safe_load(cleaned_content)

    if load_step is None:
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(ckpt_dir))[-1]
    
    ckpt = torch.load(ckpt_dir / f'step-{load_step:09d}.ckpt', map_location=device)

    reni_model_dict = {}
    for key in ckpt['pipeline'].keys():
        if key.startswith('_model.'):
            reni_model_dict[key[7:]] = ckpt['pipeline'][key]
    
    config_path = load_dir / 'config.yml'
    with open(config_path, 'r') as f:
        content = f.read()
        config = clean_and_load_yaml(content)
    
    if model_type == "RENIGon":
        print("using ReniGON")
        model_config = RENIGon.config
        if "field.train_mu" in reni_model_dict:
            del reni_model_dict["field.train_mu"]
        if "field.train_logvar" in reni_model_dict:
            del reni_model_dict["field.train_logvar"]
        if "field.eval_mu" in reni_model_dict:
            del reni_model_dict["field.eval_mu"]
        if "field.eval_logvar" in reni_model_dict:
            del reni_model_dict["field.eval_logvar"]
    elif model_type == "RENI":
        print("using Reni++")
        model_config = RENIField.config
        model_config.pipeline.model.field.old_implementation = config['pipeline']['model']['field']['old_implementation']
    elif model_type == "VENI":
        print("using VENI")
        model_config = VENI.config
    else:
        raise ValueError("not a valid model_type")

    # for RENI Dataparser, this has to be setup
    model_config.pipeline.datamanager.data = datapath
    model_config.pipeline.datamanager.dataparser.data = datapath

    
    # TODO maybe make this configurable:
    maskpath = os.path.join(datapath, "masks")

    model_config.pipeline.datamanager.dataparser.convert_to_ldr = config['pipeline']['datamanager']['dataparser']['convert_to_ldr']
    model_config.pipeline.datamanager.dataparser.convert_to_log_domain = config['pipeline']['datamanager']['dataparser']['convert_to_log_domain']
    if config['pipeline']['datamanager']['dataparser']['eval_mask_path'] is not None:
        print(config['pipeline']['datamanager']['dataparser']['eval_mask_path'])
        eval_mask_path = Path(os.path.join(maskpath, config['pipeline']['datamanager']['dataparser']['eval_mask_path'][-1]))
        model_config.pipeline.datamanager.dataparser.eval_mask_path = eval_mask_path
    else:
        model_config.pipeline.datamanager.dataparser.eval_mask_path = None
    if config['pipeline']['datamanager']['dataparser']['min_max_normalize'].__class__ == list:
        model_config.pipeline.datamanager.dataparser.min_max_normalize = tuple(config['pipeline']['datamanager']['dataparser']['min_max_normalize'])
    else:
        model_config.pipeline.datamanager.dataparser.min_max_normalize = config['pipeline']['datamanager']['dataparser']['min_max_normalize']
    model_config.pipeline.datamanager.dataparser.augment_with_mirror = config['pipeline']['datamanager']['dataparser']['augment_with_mirror']

    if "full_image_per_batch" in config['pipeline']['datamanager']['pixel_sampler']:
        model_config.pipeline.datamanager.pixel_sampler.full_image_per_batch = config['pipeline']['datamanager']['pixel_sampler']['full_image_per_batch']
        model_config.pipeline.datamanager.pixel_sampler.images_per_batch = config['pipeline']['datamanager']['pixel_sampler']['images_per_batch']

    model_config.pipeline.model.loss_inclusions = config['pipeline']['model']['loss_inclusions']
    model_config.pipeline.model.field.conditioning = config['pipeline']['model']['field']['conditioning']
    model_config.pipeline.model.field.invariant_function = config['pipeline']['model']['field']['invariant_function']
    model_config.pipeline.model.field.equivariance = config['pipeline']['model']['field']['equivariance']
    model_config.pipeline.model.field.axis_of_invariance = config['pipeline']['model']['field']['axis_of_invariance']
    model_config.pipeline.model.field.positional_encoding = config['pipeline']['model']['field']['positional_encoding']
    model_config.pipeline.model.field.encoded_input = config['pipeline']['model']['field']['encoded_input']
    model_config.pipeline.model.field.latent_dim = config['pipeline']['model']['field']['latent_dim']
    model_config.pipeline.model.field.hidden_features = config['pipeline']['model']['field']['hidden_features']
    model_config.pipeline.model.field.hidden_layers = config['pipeline']['model']['field']['hidden_layers']
    model_config.pipeline.model.field.mapping_layers = config['pipeline']['model']['field']['mapping_layers']
    model_config.pipeline.model.field.mapping_features = config['pipeline']['model']['field']['mapping_features']
    model_config.pipeline.model.field.num_attention_heads = config['pipeline']['model']['field']['num_attention_heads']
    model_config.pipeline.model.field.num_attention_layers = config['pipeline']['model']['field']['num_attention_layers']
    model_config.pipeline.model.field.output_activation = config['pipeline']['model']['field']['output_activation']
    model_config.pipeline.model.field.last_layer_linear = config['pipeline']['model']['field']['last_layer_linear']
    model_config.pipeline.model.loss_coefficients = config['pipeline']['model']['loss_coefficients']

    if model_type!="VENI":
        model_config.pipeline.model.field.trainable_scale = config['pipeline']['model']['field']['trainable_scale']

    if model_type!="RENI":
        if "input_linear" not in config["pipeline"]["model"]["field"]:
            model_config.pipeline.model.field.input_linear = False
        else: 
            model_config.pipeline.model.field.input_linear = config["pipeline"]["model"]["field"]["input_linear"]
        
        for k,v in model_config.pipeline.model.loss_coefficients.items():
            scheduler = ConstantLossWeightConfig(weight=1)
            if isinstance(v, float):
                scheduler = ConstantLossWeightConfig(weight=v)
            elif "weight" in v.keys():
                scheduler = ConstantLossWeightConfig(weight=v["weight"])
            elif "start_weight" in v.keys():
                scheduler = LinearLossWeightSchedulerConfig(start_weight=v["start_weight"], end_weight=v["end_weight"], start_step=v["start_step"], end_step=v["end_step"])
            model_config.pipeline.model.loss_coefficients[k] = scheduler
    
    if model_type=="VENI":
        model_config.pipeline.model.encoder.pooling = config['pipeline']['model']["encoder"]['pooling']
        model_config.pipeline.model.encoder.lernable_cls = config['pipeline']['model']["encoder"]['learnable_cls']
        model_config.pipeline.model.encoder.invariant_axes = config['pipeline']['model']["encoder"]['invariant_axes']
        model_config.pipeline.model.encoder.reduce_feature_to_coord = config['pipeline']['model']["encoder"]['reduce_feature_to_coord']
        model_config.pipeline.model.encoder.transformer_type = config['pipeline']['model']["encoder"]['transformer_type']
        model_config.pipeline.model.encoder.inner_dim = config['pipeline']['model']["encoder"]['inner_dim']
        model_config.pipeline.model.encoder.dim_head = config['pipeline']['model']["encoder"]['dim_head']
        model_config.pipeline.model.encoder.heads = config['pipeline']['model']["encoder"]['heads']
        model_config.pipeline.model.encoder.depth = config['pipeline']['model']["encoder"]['depth']
        model_config.pipeline.model.encoder.project_in_type = config['pipeline']['model']["encoder"]['project_in_type']
        model_config.pipeline.model.encoder.project_out_type = config['pipeline']['model']["encoder"]['project_out_type']
        if "variational" in config["pipeline"]["model"]["encoder"]:
            model_config.pipeline.model.encoder.variational = config['pipeline']['model']["encoder"]['variational']
        else:
            model_config.pipeline.model.encoder.variational = False
        if "mirror" in config["pipeline"]["model"]["encoder"]:
            model_config.pipeline.model.encoder.mirror = config['pipeline']['model']["encoder"]['mirror']
        else:
            model_config.pipeline.model.encoder.mirror = "none"


        model_config.pipeline.model.patch_size = config['pipeline']['model']["patch_size"]
        model_config.pipeline.model.sampling = config['pipeline']['model']["sampling"]
        model_config.pipeline.model.samples_per_img = config['pipeline']['model']["samples_per_img"]
        model_config.pipeline.model.fov = config['pipeline']['model']["fov"]

    if "test_mode" in config["pipeline"]:
        model_config.pipeline.test_mode = config['pipeline']['test_mode']
    else:
        model_config.pipeline.test_mode = "test"

    if "multi_scale_derivative_loss" not in model_config.pipeline.model.loss_inclusions:
        model_config.pipeline.model.loss_inclusions["multi_scale_derivative_loss"] = False
    if "gon_loss" not in model_config.pipeline.model.loss_inclusions:
        model_config.pipeline.model.loss_inclusions["gon_loss"] = False
    if "masked_mse_loss" not in model_config.pipeline.model.loss_inclusions:
        model_config.pipeline.model.loss_inclusions["masked_mse_loss"] = False


    if custom_division_factor:
        model_config.pipeline.model.division_factor = custom_division_factor
    elif "division_factor" in config["pipeline"]:
        model_config.pipeline.model.division_factor = config['pipeline']['division_factor']
    elif "division_factor" in config["pipeline"]["model"]:
        model_config.pipeline.model.division_factor = config['pipeline']['model']['division_factor']
    else:
        model_config.pipeline.model.division_factor = 1 # no division if not defined.

    if "variational" in config['pipeline']['model']['field']:
        model_config.pipeline.model.field.variational = config['pipeline']['model']['field']['variational']
    else:
        model_config.pipeline.model.field.variational = variational


    test_mode = model_config.pipeline.test_mode

    if model_only:
        model = model_config.pipeline.model.setup(
            metadata=metadata, 
            scene_box=None, 
            num_train_data=num_train_data, 
            num_eval_data=num_eval_data
        )
        model.to(device)   
        if load_state:
            model.load_state_dict(reni_model_dict, strict=False)
        model.eval()
        return model, model_config.pipeline
 

    pipeline = model_config.pipeline.setup(
      device=device,
      test_mode=test_mode,
      world_size=world_size,
      local_rank=local_rank,
      grad_scaler=None,
    )

    datamanager = pipeline.datamanager

    model = pipeline.model

    model.to(device)
    if load_state:
        model.load_state_dict(reni_model_dict)
    model.eval()

    return pipeline, datamanager, model
