import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
import torch

# new function
def adjust_pos_neg_percentage(value, pct):
    is_positive = torch.rand(1).item() < pct
    return abs(value) if is_positive else -abs(value)

def change_latent_pos_neg_percentage(tensor, target_pos_percentage=0.5):
    flat_tensor = tensor.view(-1)
    flat_tensor_len = len(flat_tensor)
    
    num_pos = (flat_tensor > 0).sum().item()
    num_neg = (flat_tensor < 0).sum().item()
    current_pos_percentage = num_pos / flat_tensor_len
    current_neg_percentage = num_neg / flat_tensor_len

    print("Percentage of positive and negative values before adjustment:")
    print("Pos:", current_pos_percentage, "Neg:", current_neg_percentage)
    
    tensor.apply_(lambda x: adjust_pos_neg_percentage(x, target_pos_percentage))
    
    # Print the final percentage of positive and negative values after adjustment
    final_pos_percentage = (flat_tensor > 0).sum().item() / flat_tensor_len
    final_neg_percentage = (flat_tensor < 0).sum().item() / flat_tensor_len
    print("Percentage of positive and negative values after adjustment:")
    print("Positive:", final_pos_percentage, "Negative:", final_neg_percentage)

    return tensor


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
    
    init_latents_w_adjusted1 = change_latent_pos_neg_percentage(init_latents_w, target_pos_percentage=1)
    print(init_latents_w_adjusted1)
    init_latents_w_adjusted2 = change_latent_pos_neg_percentage(init_latents_w, target_pos_percentage=0)
    print(init_latents_w_adjusted2)
    init_latents_w_adjusted3 = change_latent_pos_neg_percentage(init_latents_w, target_pos_percentage=0.3235)
    print(init_latents_w_adjusted3)
    init_latents_w_adjusted4 = change_latent_pos_neg_percentage(init_latents_w, target_pos_percentage=0.749)
    print(init_latents_w_adjusted4)
