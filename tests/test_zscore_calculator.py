import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import sys
import os
import torch
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *

# New function
def adjust_pos_neg_percentage(value, pct, z):
    is_positive = torch.rand(1).item() < pct
    return abs(value) - z if is_positive else -abs(value) - z

def change_latent_pos_neg_percentage(tensor, target_pos_percentage=0.5):
    flat_tensor = tensor.view(-1)
    new_tensor = torch.empty_like(flat_tensor)
    new_tensor = new_tensor.view(tensor.shape)

    return new_tensor


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base',
        scheduler=scheduler,
        torch_dtype=torch.float32,
        revision='fp16',
    )
    pipe = pipe.to(device)
  
    init_latents_w = pipe.get_random_latents()
    init_latents_w_adjusted1 = change_latent_pos_neg_percentage(torch.clone(init_latents_w), target_pos_percentage=0.55)
    # Calculate z-score
    """
    mean_latents_w = init_latents_w_adjusted1.mean()
    std_latents_w = init_latents_w_adjusted1.std()
    z_score_latents_w = (init_latents_w_adjusted1 - mean_latents_w) / std_latents_w
    """
    mean_latents_w = init_latents_w.mean()
    std_latents_w = init_latents_w.std()
    z_score_latents_w = (init_latents_w - mean_latents_w)/std_latents_w
    
    # print(f"Z-score of the original latent: {z_score_latents_w}")
    mean_z_score_latents_w = z_score_latents_w.mean().item()
    
    print(f"Mean Z-score of the original latent: {mean_z_score_latents_w}")