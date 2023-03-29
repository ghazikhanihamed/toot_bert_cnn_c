#!/bin/bash -l

#$ -N fixname
#$ -cwd
#$ -m bea
#$ -l m_mem_free=32G,g=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python fix_names.py

deactivate

