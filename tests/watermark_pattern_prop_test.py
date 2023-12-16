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
import torch
# test eval_watermark

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
    
    # reverse img with watermarking
    img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
    image_latents_w = pipe.get_image_latents(img_w, sample=False)

    reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
    )


    # eval
    no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)
        
    # functions added in optim_utils for debugging, but failed
    """
    # used to change tensor size for debugging
def resize_tensor(tensor, target_shape):
    input_dims = len(tensor.shape)
    target_dims = len(target_shape)

    # Handle 2D tensor separately
    if input_dims == 2 and target_dims == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Convert 2D to 4D
        target_shape = (1, 1) + target_shape  # Adjust target shape for 4D

    resized_tensor = torch.nn.functional.interpolate(
        tensor, size=target_shape[-2:],
        mode='bilinear', align_corners=False
    )

    # Squeeze dimensions if needed
    if input_dims == 2 and target_dims == 2:
        resized_tensor = resized_tensor.squeeze(0).squeeze(0)  # Convert 4D to 2D

    return resized_tensor
    """
    """
      gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
      #  gt_patch = resize_tensor(gt_patch, gt_init.shape[-2:]) # New change
        gt_patch_tmp = copy.deepcopy(gt_patch)
    """