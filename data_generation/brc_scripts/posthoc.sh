#!/bin/bash
# Job name:
#SBATCH --job-name=posthoc
#
# Account:
#SBATCH --account=co_rail
#
# Partition:
#SBATCH --partition=savio4_gpu
#
# Request node(s):
#SBATCH --nodes=1
#
#SBATCH --ntasks-per-node=1
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=8
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
#SBATCH --gres=gpu:A5000:1

singularity exec --writable-tmpfs --nv --env HF_HOME=/global/scratch/users/riadoshi/vla/cache/ \
    /global/home/groups/co_rail/verityw/code_img.sif \
    python /global/scratch/users/riadoshi/vla/post_hoc.py --dataset "$1"
