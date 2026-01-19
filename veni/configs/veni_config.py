"""
VENI configuration file.
"""
from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig, MachineConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig, ExponentialDecaySchedulerConfig
from nerfstudio.data.pixel_samplers import PatchPixelSampler, PatchPixelSamplerConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager

from reni.data.dataparsers.reni_dataparser import RENIDataParserConfig
from reni.data.datasets.reni_dataset import RENIDataset
from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig

from veni.models.veni_model import VENIModelConfig
from veni.model_components.loss_schedulers import LinearLossWeightSchedulerConfig, ConstantLossWeightConfig, SigmoidLossWeightSchedulerConfig
from veni.pipelines.veni_pipeline import VENIPipelineConfig
from veni.illumination_fields.veni_illumination_field import VENIFieldConfig
from veni.illumination_fields.veni_encoder import VENIEncoderConfig
from veni.data.veni_pixel_sampler import VENIEquirectangularPixelSamplerConfig


images_per_batch = 5
total_image_steps = 200000

VENI = MethodSpecification(
    config=TrainerConfig(
        method_name="veni",
        experiment_name="veni-test",
        machine=MachineConfig(),
        steps_per_eval_image=2000//images_per_batch,
        steps_per_eval_batch=2000//images_per_batch,
        steps_per_save=5000//images_per_batch,
        save_only_latest_checkpoint=True,
        steps_per_eval_all_images=2000//images_per_batch,
        max_num_iterations=total_image_steps//images_per_batch+1,
        mixed_precision=False,
        pipeline=VENIPipelineConfig( 
            test_mode='test', # Change Testmode for different testset
            datamanager=VanillaDataManagerConfig(
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
            ),
            model=VENIModelConfig(
                field=VENIFieldConfig(
                    conditioning="Attention",
                    invariant_function="VN",
                    equivariance="SO2",
                    axis_of_invariance="z",  # Nerfstudio world space is z-up # old reni implementation was y-up
                    positional_encoding="NeRF",
                    encoded_input="Directions",  # "InvarDirection", "Directions", "Conditioning", "Both", "None"
                    latent_dim=100,  # N for a latent code size of (N x 3) # 9, 36, 49, 100 (for paper sizes)
                    hidden_features=128,  # ALL
                    hidden_layers=9,  # SIRENs
                    mapping_layers=5,  # FiLM MAPPING NETWORK
                    mapping_features=128,  # FiLM MAPPING NETWORK
                    num_attention_heads=8,  # TRANSFORMER
                    num_attention_layers=6,  # TRANSFORMER
                    output_activation="None",  # ALL
                    last_layer_linear=True,  # SIRENs
                ),
                encoder=VENIEncoderConfig(
                    pooling = "cls",
                    learnable_cls = True,
                    invariant_axes = {2},
                    reduce_feature_to_coord = "late_learned",
                    transformer_type = "VN",
                    inner_dim = 256,
                    dim_head = 32,
                    heads = 8,
                    depth = 6,
                    project_in_type = "SOx",
                    project_out_type = "SOx",
                    variational = True,
                    mirror = "early",
                ),
                patch_size = 512,
                sampling = "line_sampling",
                samples_per_img = 64,
                fov = 33,
                loss_coefficients={
                    "log_mse_loss": ConstantLossWeightConfig(weight=1.0),
                    "hdr_mse_loss": ConstantLossWeightConfig(weight=1.0),
                    "ldr_mse_loss": ConstantLossWeightConfig(weight=1.0),
                    "cosine_similarity_loss": ConstantLossWeightConfig(weight=1.0),
                    "scale_inv_loss": ConstantLossWeightConfig(weight=1.0),
                    "kld_loss": LinearLossWeightSchedulerConfig(start_weight=0.01,end_weight=0.01,start_step=1, end_step=2),
                    "multi_scale_derivative_loss": ConstantLossWeightConfig(weight=0.5),
                },
                loss_inclusions={
                    "log_mse_loss": False,
                    "hdr_mse_loss": False,
                    "ldr_mse_loss": False,
                    "cosine_similarity_loss": True,
                    "kld_loss": True,
                    "scale_inv_loss": True, 
                    "multi_scale_derivative_loss": True,
                },
            ),
        ),
        optimizers={
            "field": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-8, max_norm=0.5), # For finetuning: lr=1e-5
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=1000//images_per_batch, learning_rate_alpha=0.05, max_steps=total_image_steps//images_per_batch+1),
            },
            "encoder": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-8, max_norm=0.5), # For finetuning lr=1e-4
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=1000//images_per_batch, learning_rate_alpha=0.05, max_steps=total_image_steps//images_per_batch+1),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for Rotation-Equivariant Natural Illumination Fields.",
)

