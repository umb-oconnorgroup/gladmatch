#!/bin/bash
#$ -q all.q
#$ -P toconnor-lab
#$ -o gladmatch.log
#$ -N gladmatch
#$ -l mem_free=10G
script_path=$1
config_path=$2
query_path=$3
result_path=$4
n=$5
exclude_phs=$6
self_described=$7
export PATH=/local/devel/glad/miniconda3/envs/gladmatch/bin:$PATH
export LD_LIBRARY_PATH=/local/devel/glad/miniconda3/envs/gladmatch/lib:$LD_LIBRARY_PATH
python3 ${script_path} ${config_path} ${query_path} ${result_path} -n ${n} ${self_described} --exclude-phs ${exclude_phs}
