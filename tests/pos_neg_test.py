import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch
import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)
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
    """
    num_pos = (flat_tensor > 0).sum().item()
    num_neg = (flat_tensor < 0).sum().item()
    current_pos_percentage = num_pos / flat_tensor_len
    current_neg_percentage = num_neg / flat_tensor_len

    print("Percentage of positive and negative values before adjustment:")
    print("Pos:", current_pos_percentage, "Neg:", current_neg_percentage)
    """
    # tensor.apply_ is not working
    # tensor.apply_(lambda x: adjust_pos_neg_percentage(x, target_pos_percentage))
    # try to create a new tensor
    new_tensor = torch.tensor([adjust_pos_neg_percentage(x.item(), target_pos_percentage) for x in flat_tensor], dtype=tensor.dtype)
    new_tensor = new_tensor.view(tensor.shape)
    new_flat_tensor = new_tensor.view(-1)
    # Print the final percentage of positive and negative values after adjustment
    final_pos_percentage = (new_flat_tensor > 0).sum().item() / len(new_flat_tensor)
    final_neg_percentage = (new_flat_tensor < 0).sum().item() / len(new_flat_tensor)
    print("Percentage of positive and negative values after adjustment:")
    print("Positive:", final_pos_percentage, "Negative:", final_neg_percentage)

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
    print(init_latents_w)
    print("1")
    init_latents_w_adjusted1 = change_latent_pos_neg_percentage(torch.clone(init_latents_w), target_pos_percentage=1)
    print(init_latents_w_adjusted1)
    print("2")
    init_latents_w_adjusted2 = change_latent_pos_neg_percentage(torch.clone(init_latents_w), target_pos_percentage=0)
    #print(init_latents_w_adjusted2)
    print("3")
    init_latents_w_adjusted3 = change_latent_pos_neg_percentage(torch.clone(init_latents_w), target_pos_percentage=0.3235)
    #print(init_latents_w_adjusted3)
    print("4")
    init_latents_w_adjusted4 = change_latent_pos_neg_percentage(torch.clone(init_latents_w), target_pos_percentage=0.749)
    #print(init_latents_w_adjusted4)
    print("5 and 6")
    init_latents_w_adjusted5 = change_latent_pos_neg_percentage(torch.clone(init_latents_w), target_pos_percentage=0.3)
    init_latents_w_adjusted6 = change_latent_pos_neg_percentage(torch.clone(init_latents_w), target_pos_percentage=0.3)