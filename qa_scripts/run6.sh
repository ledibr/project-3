#!/bin/bash

#SBATCH --job-name=cosi231a-pa3
#SBATCH --output=logs/run_%j_lora_rank_large.out
#SBATCH --error=logs/run_%j_lora_rank_large.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1024
#SBATCH --gres=gpu:1

cd ~/cosi231/project-3
OUTPUT_ROOT=~/cosi231/project-3/output

OUTPUT_DIR=$OUTPUT_ROOT/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTPUT_DIR"

uv run main.py --task qa --tuning True --rank 32