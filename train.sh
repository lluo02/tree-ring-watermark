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
python run_tree_ring_watermark.py --r_degree 90 --run_name attack-tol1-90deg --w_channel 3 --w_pattern ring_tol --start 0 --end 1 --with_tracking --w_pos_ratio 0.5

