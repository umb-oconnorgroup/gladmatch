#!/bin/bash
#$ -q all.q
#$ -P toconnor-lab
#$ -o gladmatch.log
#$ -N gladmatch
#$ -l mem_free=10G
# pca_path=static/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.filtered.pca.applied_to_all.npy
# explained_variance_ratio_path=static/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.filtered.explained_variance_ratio.npy
# zarr_path=static/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.zarr
pca_path=/autofs/chib/oconnor_genomes/GLAD/F1_merged_data/gladdb/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.filtered.pca.applied_to_all.npy
explained_variance_ratio_path=/autofs/chib/oconnor_genomes/GLAD/F1_merged_data/gladdb/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.filtered.explained_variance_ratio.npy
zarr_path=/autofs/chib/oconnor_genomes/GLAD/F1_merged_data/gladdb/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.zarr
pruned_zarr_path=/autofs/chib/oconnor_genomes/GLAD/F1_merged_data/gladdb/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.zarr
demography_path=/autofs/chib/oconnor_genomes/GLAD/F1_merged_data/gladdb/data/12082021_Demographic_details_GLAD_Freeze1.txt
local_ancestry_path=/local/chib/oconnor_genomes/GLAD/F1_merged_data/imputed_above0.9/Local_ancestry_202108/Outputs_Phased_202111_rfmix_8G/GLAD_F1_Phased_rfmix_G8_202111_chr1.msp.tsv
# local_ancestry_path=/local/chib/oconnor_genomes/GLAD/F1_merged_data/imputed_above0.9/Local_ancestry_202108/Outputs_Phased_202201_rfmix_8G/GLAD_F1_Phased_rfmix_G8_202201_chr1.msp.tsv
script_path=$1
query_path=$2
result_path=$3
n=$4
exclude_phs=$5
self_described=$6
export PATH=/local/devel/glad/miniconda3/envs/gladmatch/bin:$PATH
export LD_LIBRARY_PATH=/local/devel/glad/miniconda3/envs/gladmatch/lib:$LD_LIBRARY_PATH
python3 ${script_path} ${pca_path} ${explained_variance_ratio_path} ${zarr_path} ${pruned_zarr_path} ${query_path} ${demography_path} ${local_ancestry_path} ${result_path} -n ${n} ${self_described} --exclude-phs ${exclude_phs}
