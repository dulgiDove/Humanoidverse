#!/bin/bash
source /home/k1010282/miniconda3/etc/profile.d/conda.sh
conda activate hgen
cd /mnt/c/Users/pc123/HumanoidVerse
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python humanoidverse/eval_agent.py "+checkpoint=logs/H1_Stage3/20260513_033805-H1_10dof_obstacle_v7-locomotion-h1_10dof/model_75700.pt" +headless=True +num_envs=1
