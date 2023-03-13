#!/bin/bash -l

#$ -N tunecnn
#$ -cwd
#$ -m bea
#$ -l m_mem_free=32G,g=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.7.3/default

source ~/python_path_gpu/bin/activate

python tune_cnn.py

deactivate

