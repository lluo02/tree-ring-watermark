#!/bin/bash
  
#SBATCH --job-name=treering
#SBATCH --output=treering.out.%j
#SBATCH --error=treering.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=vulcan-abhinav
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --gres=gpu:rtxa4000:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G


module load cuda/10.2.89
python run_tree_ring_watermark.py  --gaussian_blur_r 4 --run_name blur4-ring_prop55 --w_mask_shape no --w_channel 3 --w_pattern ring_prop --start 0 --end 1 --with_tracking --w_pos_ratio 0.8 --w_radius 1 --output_file blur4-ring_prop80-noreg

