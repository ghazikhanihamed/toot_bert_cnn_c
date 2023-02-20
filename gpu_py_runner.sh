#!/bin/bash -l

#$ -N evolbert_frzn
#$ -cwd
#$ -m bea
#$ -l m_mem_free=32G,g=3

export TMPDIR=~/tmp

module load python/3.7.3/default

module load cuda/9.2/default

source ~/python_path/bin/activate

python evol_bert_frozen.py 

deactivate

