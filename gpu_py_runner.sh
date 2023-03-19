#!/bin/bash -l

#$ -N tunecnn
#$ -cwd
#$ -m bea
#$ -l m_mem_free=64G,g=8

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python tune_cnn.py

deactivate

