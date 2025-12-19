#!/bin/bash

#SBATCH --job-name=cosi231a-pa3
#SBATCH --output=logs/run_%j_lora_base.out
#SBATCH --error=logs/run_%j_lora_base.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --partition=gpu48g
#SBATCH --gres=gpu:1

export HF_HOME=/data/ldial/.cache/huggingface
export HF_DATASETS_CACHE=/data/ldial/.cache/huggingface/datasets
export UV_CACHE_DIR=/data/ldial/.cache/uv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_ROOT=/data/$(whoami)/
cd ~/cosi231/project-3

OUTPUT_DIR=$OUTPUT_ROOT/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTPUT_DIR"

uv run main.py --task gen --epochs 1 --train_size 0.3 --tuning=True --batch_size 32 --rank 8