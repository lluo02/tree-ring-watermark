#!/bin/bash
  
#SBATCH --job-name=treering
#SBATCH --output=ae_sd.out.%j
#SBATCH --error=ae_sd.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=cml-scavenger
#SBATCH --partition=cml-scavenger
#SBATCH --qos=cml-scavenger
#SBATCH --gres=gpu:rtxa4000:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G


module load cuda/10.2.89
python run_tree_ring_watermark.py --run_name no_attack_pos_ratio_30 --w_channel 3 --w_pattern ring --start 0 --end 10 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k --w_pos_ratio 0.7