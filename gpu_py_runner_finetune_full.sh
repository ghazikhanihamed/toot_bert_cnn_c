#!/bin/bash -l

#$ -N savefllfntn
#$ -cwd
#$ -m bea
#$ -l m_mem_free=64G,g=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python save_representations_finetuned.py 

deactivate

