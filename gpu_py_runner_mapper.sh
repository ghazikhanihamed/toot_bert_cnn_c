#!/bin/bash -l

#$ -N mapfull
#$ -cwd
#$ -m bea
#$ -l m_mem_free=32G,g=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python map_representations_dataset_frozen_full.py

deactivate

