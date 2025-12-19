#!/bin/bash

#SBATCH --job-name=cosi231a-pa3
#SBATCH --output=logs/run_%j_base.out
#SBATCH --error=logs/run_%j_base.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1024
#SBATCH --gres=gpu:1

set UV_CACHE_DIR=/data/$(whoami)/

OUTPUT_ROOT=/data/$(whoami)/
cd ~/cosi231/project-3

OUTPUT_DIR=$OUTPUT_ROOT/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTPUT_DIR"

uv run main.py --task qa