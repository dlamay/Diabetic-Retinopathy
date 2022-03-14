#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=autoencoder_run
#SBATCH --mem=120GB
#SBATCH --ntasks=1

module load anaconda3/2022.01
module load cuda/11.0
conda create --name TF_env python=3.7 anaconda
source activate TF_env
conda install -c anaconda tensorflow-gpu -y
python train_cnn_resnet.py 
