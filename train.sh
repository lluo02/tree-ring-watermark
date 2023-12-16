#!/bin/bash
  
#SBATCH --job-name=treering
#SBATCH --output=treering.out.%j
#SBATCH --error=treering.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=cml-scavenger
#SBATCH --partition=cml-scavenger
#SBATCH --qos=cml-scavenger
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G


module load cuda/10.2.89
python run_tree_ring_watermark.py --run_name 55_prop_90deg --w_channel 3 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k --r_degree 90 --w_pos_ratio 0.55

