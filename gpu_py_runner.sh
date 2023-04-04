#!/bin/bash -l

#$ -N test_res
#$ -cwd
#$ -m bea
#$ -l m_mem_free=11G,g=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

python make_results_soa.py

deactivate

