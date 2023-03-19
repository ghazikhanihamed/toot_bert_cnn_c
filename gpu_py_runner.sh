#!/bin/bash -l

#$ -N tunecnn
#$ -cwd
#$ -m bea
#$ -l m_mem_free=32G,g=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python tune_cnn.py

deactivate

