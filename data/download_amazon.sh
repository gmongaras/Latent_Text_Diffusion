#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Mrrrowww_^w^
#SBATCH -p batch
#SBATCH -o download.out
#SBATCH --mem=500G
#SBATCH --gres=gpu:1

source ~/.bashrc
cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Text_Diffusion/
python data/Amazon_Reviews.py
