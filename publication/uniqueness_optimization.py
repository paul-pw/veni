# Optimize the latents for the figure
from helpers import models, optimize_latents_of_image, load_dataloader
from reni.utils.helpers import load_model, device
from reni.utils.utils import find_nerfstudio_project_root
import pickle
import random
import torch
from pathlib import Path
import os
import argparse

project_root = find_nerfstudio_project_root(Path(os.getcwd()))
os.chdir(project_root)


parser = argparse.ArgumentParser(prog='Uniqueness optimization')
parser.add_argument('--seed')
args = parser.parse_args()

image_to_solve = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

dataloader_mode = "test"
test_dataloader, metadata, num_train_data, num_eval_data = load_dataloader(dataloader_mode)

uniqueness_optimization_full_opt = {}
seed = int(args.seed)
for i,m in enumerate(models):
    model_type = m["model_type"]
    model, pipeline_config = load_model(Path(m["model"]), model_only=True, metadata=metadata, model_type=model_type)
   
    lr = (1e-2, 2e-3)
    if model_type == "RENI":
        lr =(1e-1, 1e-3)

    uniqueness_optimization_full_opt[m["name"]] = {}
    torch.manual_seed(seed)
    random.seed(seed)
    for j in image_to_solve:
        test_dataloader.count = j
        camera, batch = next(test_dataloader)
        latents1 = optimize_latents_of_image(batch["image"], torch.randn(1,model.field.latent_dim,3, device=device), model, camera, steps=2000, lr=lr)
        latents2 = optimize_latents_of_image(batch["image"], torch.randn(1,model.field.latent_dim,3, device=device), model, camera, steps=2000, lr=lr)
        uniqueness_optimization_full_opt[m["name"]][j] = [latents1.detach().cpu(), latents2.detach().cpu()]
        
with open(f"publication/uniqueness_optimization_full_{seed}.pkl", 'wb') as file:
    pickle.dump(uniqueness_optimization_full_opt, file)
