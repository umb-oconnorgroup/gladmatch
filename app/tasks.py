import os
import tempfile
import uuid

from celery import Celery
import dask.array as da
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

pca_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.filtered.pca.applied_to_all.npy")
explained_variance_ratio_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.filtered.explained_variance_ratio.npy")
zarr_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.zarr")
glad_pca = np.load(pca_path)
explained_variance_ratio = np.load(explained_variance_ratio_path)
glad_genotypes = da.from_zarr(os.path.join(zarr_path, "call_genotype"))
glad_chromosomes = (da.from_zarr(os.path.join(zarr_path, "variant_contig")) + 1).compute()
glad_positions = da.from_zarr(os.path.join(zarr_path, "variant_position")).compute()
#
abridged_genotypes_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static/data/Freeze1.imputed_above0.9.missing0.001.nodups_202106.sorted.biallelic.pruned_at0.5.genotypes.abridged.npy")
abridged_genotypes = np.load(abridged_genotypes_path)
#

app = Celery('tasks', backend="redis://localhost", broker="pyamqp://")

def match_query(embedding, query_embedding, metric="euclidean", weights=None):
    kwargs = {}
    if metric == "wminkowski":
        kwargs["w"] = weights
    distances = cdist(query_embedding, embedding, metric=metric, **kwargs)
    return np.array(linear_sum_assignment(distances)[1])

@app.task
def match(file_id):
    query_pca = np.load(os.path.join(tempfile.gettempdir(), file_id))
    # match_indices = match_query(glad_pca, query_pca, metric="wminkowski", weights=explained_variance_ratio)
    # match_allele_frequencies = (glad_genotypes[:, match_indices].sum(-1).sum(-1) / (glad_genotypes.shape[1] * glad_genotypes.shape[2])).compute()
    # abridged for ram limitation
    match_indices = match_query(glad_pca[:abridged_genotypes.shape[0]], query_pca, metric="wminkowski", weights=explained_variance_ratio)
    match_allele_frequencies = abridged_genotypes[:, match_indices].sum(-1).sum(-1) / (abridged_genotypes.shape[1] * abridged_genotypes.shape[2])
    result_file_id = str(uuid.uuid1())
    np.savez_compressed(os.path.join(tempfile.gettempdir(), result_file_id), chromosomes=glad_chromosomes, positions=glad_positions, minor_allele_frequencies=match_allele_frequencies)
    return result_file_id
