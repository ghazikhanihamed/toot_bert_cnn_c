#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=64G
#SBATCH -J finetune
#SBATCH -o _%x%J.out
#SBATCH --gpus=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.9.6
module load anaconda/3.2023.03
module load cuda/12.1.1

source /usr/local/pkg/anaconda/v3.2023.03/root/etc/profile.d/conda.sh
conda activate py39

nvidia-smi

python save_finetuned_models_new.py

conda deactivate
module purge
