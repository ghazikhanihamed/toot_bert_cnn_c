#!/bin/bash -l

#$ -N cnncv
#$ -cwd
#$ -m bea
#$ -l m_mem_free=32G,g=4

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python -m torch.distributed.launch --nproc_per_node=4 cnn_cv.py 

deactivate

