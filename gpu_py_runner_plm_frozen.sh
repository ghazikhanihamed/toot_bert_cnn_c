#!/encs/bin/tcsh

#$ -N plm_frzn
#$ -cwd
#$ -m bea
#$ -l gpu=2

setenv TMPDIR /nfs/speed-scratch/h_ghazik/tmp
setenv TRANSFORMERS_CACHE /nfs/speed-scratch/h_ghazik/tmp

module load pytorch/1.10.0/GPU/default

source /nfs/speed-scratch/h_ghazik/python_gpu_path/bin/activate.csh

python save_frozen_representations.py

deactivate
