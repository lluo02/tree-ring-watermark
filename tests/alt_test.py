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
positive = True
tol = 0.01
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
  init_latents_w.apply_(lambda x: alt(x))
  print(init_latents_w)


