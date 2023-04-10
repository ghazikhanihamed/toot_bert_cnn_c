#!/bin/bash -l

#$ -N cnn_test
#$ -cwd
#$ -m bea
#$ -l m_mem_free=64G,g=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python make_results_soa_cnn.py

deactivate

