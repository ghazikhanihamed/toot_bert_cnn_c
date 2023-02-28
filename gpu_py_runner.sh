#!/bin/bash -l

#$ -N plm_frzn
#$ -cwd
#$ -m bea
#$ -l m_mem_free=300G,g=2

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.7.3/default

source ~/python_path_gpu/bin/activate

python save_representations.py 

deactivate

