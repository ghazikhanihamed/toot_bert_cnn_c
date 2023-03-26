#!/bin/bash -l

#$ -N fullfntn
#$ -cwd
#$ -m bea
#$ -l m_mem_free=64G,g=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python save_finetuned_models_full.py 

deactivate

