import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
import matplotlib.pyplot as plt

import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *

positive = True
def alt(x):
    global positive
    if positive:
        positive = not positive 
        if x < 0:
            return -x
        return x
    else:
        positive = not positive 
        if x > 0:
            return -x
        return x

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
  
  
  flat_tensor = init_latents_w.view(-1)
  new_tensor = torch.tensor([alt(round(x.item(), 2)) for x in flat_tensor], dtype=init_latents_w.dtype)
  new_tensor = new_tensor.view(init_latents_w.shape)
  init_latents_w = new_tensor.to(device)
  
  print("new")
  print(init_latents_w)

  t = np.squeeze(init_latents_w) # you can give axis attribute if you wanna squeeze in specific dimension 


