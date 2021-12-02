#!/bin/csh
#$ -l gpu_card=1 #
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N ernie_base  # Specify job name
module load python/3.7.3
cd /afs/crc.nd.edu/user/a/apoudel/projects/MLProject
source ./venv/bin/activate
# $data_dir=/scratch365/jlin6/data/git_data/clean/run_2
# $out_dir=~/projects/SEBert/git_split_data_final/
# $py_file=./scripts/data_process/split_dataset.py
python ./data_process/train.py 
