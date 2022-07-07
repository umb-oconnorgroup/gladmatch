import datetime
import argparse
import re

from numba import jit
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import chi2
import sgkit as sg
import yaml


NUM_WORKERS = 4
MIN_QUERY = 10

@jit(nopython=True)
def vectorized_chi2(counts, category_totals, expected_freq, chi2_values):
    for i in range(counts.shape[0]):
        chi2_values[i] = 0
        for j in range(counts.shape[1]):
            category_totals[j] = np.sum(counts[i, j])
        total = np.sum(category_totals)
        for j in range(counts.shape[1]):
            for k in range(counts.shape[2]):
                expected_freq[j, k] = np.sum(counts[i, :, k]) * category_totals[j] / total
                chi2_values[i] += np.power(counts[i, j, k] - expected_freq[j, k], 2) / expected_freq[j, k]
    return chi2_values

def evaluate_match(query_genotype_counts, genotype_counts):
    contingency_tables = np.concatenate([query_genotype_counts, genotype_counts], axis=1).reshape(-1, 2, 3)
    # filter out snps for which one of the genotype categories is empty
    contingency_tables = contingency_tables[contingency_tables.sum(1).all(1)]
    category_totals, expected_freq, chi2_values = np.zeros((contingency_tables.shape[1],), dtype=int), np.zeros((contingency_tables.shape[1], contingency_tables.shape[2]), dtype=float), np.zeros((contingency_tables.shape[0],), dtype=float)
    chi2_values = vectorized_chi2(contingency_tables, category_totals, expected_freq, chi2_values)
    chi2_values = chi2_values[~np.isnan(chi2_values)]
    return np.median(chi2_values) / chi2.ppf(0.5, 2)

def seed_match(distances, n_matches):
    match_indices = np.array(linear_sum_assignment(distances)[1])
    distances[:, match_indices] = 1e10
    while len(match_indices) != n_matches:
        if len(match_indices) > n_matches:
            n_keep_from_latest = n_matches - len(match_indices) + distances.shape[0]
            keep = np.concatenate([np.arange(len(match_indices) - distances.shape[0]), len(match_indices) - distances.shape[0] + np.random.permutation(distances.shape[0])[:n_keep_from_latest]])
            match_indices = match_indices[keep]
        else:
            new_match_indices = np.array(linear_sum_assignment(distances)[1])
            distances[:, new_match_indices] = 1e10
            match_indices = np.concatenate([match_indices, new_match_indices])
    return np.sort(match_indices)

def match_query(query_embedding, query_genotype_counts, embedding, dataset, n_matches=-1, exclude_indices=None, metric="euclidean", weights=None):
    kwargs = {}
    if weights is not None:
        kwargs["w"] = weights
    # could speed up by directly searching for nearest neighbors instead of first computing distances
    distances = cdist(query_embedding, embedding, metric=metric, **kwargs)
    if exclude_indices is not None and len(exclude_indices) > 0:
        distances[:, exclude_indices] = 1e10
    if n_matches == -1:
        n_matches = query_embedding.shape[0]
    match_indices = seed_match(np.copy(distances), n_matches)
    genotypes = dataset.call_genotype[:, match_indices].sum("ploidy").compute(num_workers=NUM_WORKERS).to_numpy()
    genotype_counts = np.zeros((dataset.call_genotype.sizes["variants"], 3), dtype=int)
    for i in range(0, 3):
        genotype_counts[:, i] = (genotypes == i).sum(1)
    current_quality_score = evaluate_match(query_genotype_counts, genotype_counts)
    # initialize best quality score to greedy approach
    best_quality_score = current_quality_score
    best_match_indices = match_indices
    print("Greedy lambda: {}".format(current_quality_score))
    n_neighbors = 5
    if n_neighbors * query_embedding.shape[0] < n_matches:
        n_neighbors = int(n_matches // query_embedding.shape[0]) + 1
    n_swap_indices = n_matches // 20 if n_matches >= 20 else 1
    n_iterations = 200
    n_start_samples = 20
    candidate_indices, counts = np.unique(distances.argsort(1)[:, :n_neighbors], return_counts=True)
    while len(candidate_indices) < n_matches:
        n_neighbors += 1
        candidate_indices, counts = np.unique(distances.argsort(1)[:, :n_neighbors], return_counts=True)
        if n_neighbors > int(n_matches // query_embedding.shape[0]) * 5:
            raise ValueError("Cannot find sufficient matches with reasonable number of neighbors.")
    # sample starting set
    candidate_genotypes = dataset.call_genotype[:, candidate_indices].sum("ploidy").compute(num_workers=NUM_WORKERS).to_numpy()
    sampled_starts = []
    for i in range(n_start_samples):
        match_index_indices = np.random.choice(np.arange(len(candidate_indices)), size=(n_matches,), replace=False, p=counts / counts.sum())
        genotype_counts = np.zeros((dataset.call_genotype.sizes["variants"], 3), dtype=int)
        for i in range(0, 3):
            genotype_counts[:, i] = (candidate_genotypes[:, match_index_indices] == i).sum(1)
        current_quality_score = evaluate_match(query_genotype_counts, genotype_counts)
        sampled_starts.append((current_quality_score, match_index_indices))
    # select start set and set initial temperature
    sampled_starts = sorted(sampled_starts, key=lambda x: x[0])
    current_quality_score, match_index_indices = sampled_starts[0]
    if current_quality_score < best_quality_score:
        best_quality_score = current_quality_score
        best_match_indices = candidate_indices[match_index_indices]
    quality_score_std = np.std([start[0] for start in sampled_starts])
    initial_temperature = -quality_score_std / np.log(quality_score_std)
    print("Initial temperature: {}".format(initial_temperature))
    print("Simulated annealing init lambda: {}".format(current_quality_score))
    for iteration_index in range(n_iterations):
        counts_copy = np.copy(counts)
        counts_copy[match_index_indices] = 0
        swap_index_indices = np.random.choice(np.arange(len(match_index_indices)), size=(n_swap_indices,), replace=False)
        swap_indices = match_index_indices[swap_index_indices]
        counts_copy[swap_indices] = counts[swap_indices]
        replacement_index_indices = np.random.choice(np.arange(len(candidate_indices)), size=(n_swap_indices,), replace=False, p=counts_copy / counts_copy.sum())
        mask = np.ones(len(match_index_indices), dtype=bool)
        mask[swap_index_indices] = False
        proposed_match_index_indices = np.concatenate([match_index_indices[mask], replacement_index_indices])
        genotype_counts = np.zeros((dataset.call_genotype.sizes["variants"], 3), dtype=int)
        for i in range(0, 3):
            genotype_counts[:, i] = (candidate_genotypes[:, proposed_match_index_indices] == i).sum(1)
        quality_score = evaluate_match(query_genotype_counts, genotype_counts)
        print("Simulated annealing iteration lambda: {}".format(quality_score))
        temperature = initial_temperature / (1 + np.log(1 + iteration_index))
        print(np.exp(-(quality_score - current_quality_score) / temperature))
        if quality_score < current_quality_score or np.exp(-(quality_score - current_quality_score) / temperature) > np.random.rand():
            print("ACCEPT")
            match_index_indices = proposed_match_index_indices
            current_quality_score = quality_score
            if current_quality_score < best_quality_score:
                best_quality_score = current_quality_score
                best_match_indices = candidate_indices[match_index_indices]
        else:
            print("REJECT")
        print("Best lambda: {}".format(best_quality_score))
    return sorted(best_match_indices), best_quality_score

def build_demographic_data(demography_path, ds):
    columns = ["Cohort", "PHS", "Country", "State", "City", "A", "B", "C", "Sex", "SelfDescribedStatus", "HispanicJustification"]
    demographic_df = pd.read_csv(demography_path, header=None, index_col=0, sep='\t', names=columns)
    demographic_df = demographic_df[~demographic_df.index.duplicated(keep="first")]
    sample_ids = ds.sample_id.compute(num_workers=NUM_WORKERS).to_numpy().tolist()
    demographic_sample_ids = demographic_df.index.to_numpy().tolist()
    demographic_df.drop(set(demographic_sample_ids).difference(set(sample_ids)), inplace=True)
    missing_sample_ids = set(sample_ids).difference(set(demographic_sample_ids))
    if len(missing_sample_ids) > 0:
        demographic_df = pd.concat([demographic_df, pd.DataFrame(np.array(['Unknown'], dtype=object).repeat(len(missing_sample_ids)).reshape(-1, 1).repeat(len(demographic_df.columns), axis=1), index=list(missing_sample_ids), columns=demographic_df.columns)])
    missing_sample_indices = [i for i in range(len(sample_ids)) if sample_ids[i] in missing_sample_ids]
    demographic_df.fillna('Unknown', inplace=True)
    demographic_df = demographic_df.reindex(sample_ids)
    demographic_df["Sample ID"] = demographic_df.index
    demographic_df.reset_index(drop=True, inplace=True)
    return demographic_df, missing_sample_indices

def get_ancestry_counts(local_ancestry_path, local_ancestry_references_path, ds, match_ids):
    positions = ds.variant_position.compute(num_workers=NUM_WORKERS).to_numpy()
    variant_contigs = ds.variant_contig.compute(num_workers=NUM_WORKERS).to_numpy()
    with open(local_ancestry_path, "r") as f:
        populations = f.readline().strip("\n").split(": ", 1)[1].split("\t")
    population_codes = [tuple(population_entry.split("=")) for population_entry in populations]
    reverse_population_codes = {population: int(code) for population, code in population_codes}
    population_codes = {int(code): population for population, code in population_codes}
    ancestry_counts = np.zeros((ds.call_genotype.sizes["variants"], len(population_codes)), dtype=int)
    references = {}
    with open(local_ancestry_references_path, "r") as f:
        for line in f.readlines():
            reference_id, population = line.split()
            references[reference_id] = reverse_population_codes[population]
    non_reference_match_ids = []
    for match_id in match_ids:
        if match_id in references:
            ancestry_counts[:, references[match_id]] += 2
        else:
            non_reference_match_ids.append(match_id)
    haploid_match_ids = [match_id + ".{}".format(i) for match_id in non_reference_match_ids for i in [0, 1]]
    for contig_i, contig in enumerate(ds.contigs):
        contig_filter = variant_contigs == contig_i
        contig_positions = positions[contig_filter]
        chr_local_ancestry_path = re.sub(r"chr\d\d?", contig, local_ancestry_path)
        windows = pd.read_csv(chr_local_ancestry_path, sep="\t", header=1, usecols=["spos", "epos"]).to_numpy()
        window_indices = np.zeros((len(contig_positions),), dtype=int)
        window_index = 0
        for pos_i in range(len(contig_positions)):
            while contig_positions[pos_i] >= windows[window_index, 1] and window_index != len(windows) - 1:
                window_index += 1
            window_indices[pos_i] = window_index
        ancestry_labels = pd.read_csv(chr_local_ancestry_path, sep="\t", header=1, usecols=haploid_match_ids).to_numpy()
        chr_ancestry_counts = np.zeros((ancestry_labels.shape[0], len(population_codes)), dtype=int)
        for code in population_codes:
            chr_ancestry_counts[:, code] = (ancestry_labels == code).sum(1)
        ancestry_counts[contig_filter] = chr_ancestry_counts[window_indices]
    return ancestry_counts, population_codes

def write_vcf(match_indices, quality_score, dataset, demographic_df, ancestry_counts, population_codes, args):
    genotypes = dataset.call_genotype[:, match_indices].sum("ploidy").compute(num_workers=NUM_WORKERS).to_numpy()
    genotype_counts = np.zeros((dataset.call_genotype.sizes["variants"], 3), dtype=int)
    for i in range(0, 3):
        genotype_counts[:, i] = (genotypes == i).sum(1)
    allele_counts = genotypes.sum(-1)
    allele_frequencies = allele_counts / (len(match_indices) * 2)
    chromosomes = [dataset.contigs[contig_index] for contig_index in dataset.variant_contig.compute(num_workers=NUM_WORKERS).to_numpy()]
    positions = dataset.variant_position.compute(num_workers=NUM_WORKERS).to_numpy()
    alleles = map(lambda x: [x[0].decode(), x[1].decode()], dataset.variant_allele.compute(num_workers=NUM_WORKERS).to_numpy())
    with open(args.result_path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##fileDate={}\n".format(datetime.date.today().strftime("%Y%m%d")))
        f.write("##source=gladdb.igs.umaryland.edu\n")
        for contig in dataset.contigs:
            f.write("##contig=<ID={}>\n".format(contig))
        f.write("##pipeline=michigan-imputationserver-1.5.7\n")
        f.write("##imputation=minimac4-1.0.2\n")
        f.write("##phasing=eagle-2.4\n")
        f.write("##r2Filter=0.9\n")
        f.write("##biallelic-filtering\n")
        # note the matching parameters
        if args.self_described:
            f.write("##match-parameter:self-described-hispanic-only")
        f.write("##match-parameter:exclude-phs={}\n".format(",".join(args.exclude_phs)))
        f.write("##match-parameter:n-matches={}\n".format(args.n))
        f.write("##match-result:n-matches={}\n".format(len(match_indices)))
        f.write("##match-result:quality-score={}\n".format(quality_score))
        sex_counts = demographic_df["Sex"].value_counts()
        f.write("##match-result:female-count={}\n".format(sex_counts["Female"]))
        f.write("##match-result:male-count={}\n".format(sex_counts["Male"]))
        f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Alternate Allele Frequency">\n')
        f.write('##INFO=<ID=AC,Number=A,Type=Int,Description="Alternate Allele Count">\n')
        f.write('##INFO=<ID=GTC,Number=A,Type=Int,Description="Genotype Counts in order of Homozygous Reference, Heterozygous, and Homozygous Alternate">\n')
        f.write('##INFO=<ID=ANC,Number=A,Type=Int,Description="Ancestry Counts in order of {}">\n'.format(", ".join([population_codes[i] for i in range(len(population_codes))])))
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chromosome, position, (major_allele, minor_allele), allele_frequency, allele_count, genotype_count, ancestry_count in zip(chromosomes, positions, alleles, allele_frequencies, allele_counts, genotype_counts, ancestry_counts):
            genotype_count_str = ",".join([str(count) for count in genotype_count.tolist()])
            ancestry_count_str = ",".join([str(count) for count in ancestry_count.tolist()])
            f.write("{}\t{}\t{}:{}:{}:{}\t{}\t{}\t.\tPASS\tAF={};AC={};GTC={};ANC={}\n".format(chromosome, position, chromosome, position, major_allele, minor_allele, major_allele, minor_allele, allele_frequency, allele_count, genotype_count_str, ancestry_count_str))

def construct_query(query_indices, dataset, pca):
    if len(query_indices) < MIN_QUERY:
        raise(ValueError("Query should be comprised of at least {} genomes".format(MIN_QUERY)))
    query_pca = pca[query_indices]
    genotypes = dataset.call_genotype[:, query_indices].sum("ploidy").compute(num_workers=NUM_WORKERS).to_numpy()
    query_gc = np.zeros((dataset.call_genotype.sizes["variants"], 3), dtype=int)
    for i in range(0, 3):
        query_gc[:, i] = (genotypes == i).sum(1)
    return query_pca, query_gc

def construct_phs_query(phs, demographic_df, dataset, pca):
    query_indices = []
    for phs in phs.split(","):
        phs = phs.strip().lower()
        if len(phs) > 0:
            if not phs.startswith("phs"):
                phs = "phs" + phs
            query_indices.extend(demographic_df.index[demographic_df["PHS"].apply(lambda x: x.lower() == phs)].tolist())
    query_indices = sorted(list(set(query_indices)))
    query_pca, query_gc = construct_query(query_indices, dataset, pca)
    return query_pca, query_gc, query_indices

def construct_cohort_query(cohort, demographic_df, dataset, pca):
    query_indices = []
    for cohort in cohort.split(","):
        cohort = cohort.strip()
        if len(cohort) > 0:
            query_indices.extend(demographic_df.groupby("Cohort").get_group(cohort).index.tolist())
    query_indices = sorted(list(set(query_indices)))
    query_pca, query_gc = construct_query(query_indices, dataset, pca)
    return query_pca, query_gc, query_indices

def construct_ids_query(ids_path, demographic_df, dataset, pca):
    query_indices = []
    with open(ids_path, "r") as f:
        ids = f.readlines()
    unique_ids = set()
    for i in ids:
        # keep only first contiguous string in line
        i = i.strip().split()[0]
        if len(i) > 0:
            unique_ids.add(i)
    query_indices.extend(demographic_df.index[demographic_df["Sample ID"].apply(lambda x: x in unique_ids)].tolist())
    query_indices = sorted(list(set(query_indices)))
    query_pca, query_gc = construct_query(query_indices, dataset, pca)
    return query_pca, query_gc, query_indices

def match(args):
    glad_pca = np.load(args.pca_path)
    explained_variance_ratio = np.load(args.explained_variance_ratio_path)
    ds = sg.load_dataset(args.zarr_path)
    pruned_ds = sg.load_dataset(args.pruned_zarr_path)
    demographic_df, exclude_indices = build_demographic_data(args.demography_path, ds)
    # if desired, filter down to only self-described latinos
    if args.self_described:
        exclude_indices.extend(demographic_df.index[demographic_df["SelfDescribedStatus"] != "Hispanic"].tolist())
    if args.exclude_phs is not None:
        args.exclude_phs = args.exclude_phs.split(",")
    else:
        args.exclude_phs = []
    if args.kgp:
        phs_values = demographic_df.PHS.unique().tolist()
        del phs_values[phs_values.index("1000Genomes")]
        args.exclude_phs.extend(phs_values)
    if args.test_phs is not None:
        query_pca, query_gc, query_indices = construct_phs_query(args.test_phs, demographic_df, pruned_ds, glad_pca)
        exclude_indices.extend(query_indices)
    elif args.test_cohort is not None:
        query_pca, query_gc, query_indices = construct_cohort_query(args.test_cohort, demographic_df, pruned_ds, glad_pca)
        exclude_indices.extend(query_indices)
    elif args.test_ids is not None:
        query_pca, query_gc, query_indices = construct_ids_query(args.test_ids, demographic_df, pruned_ds, glad_pca)
        exclude_indices.extend(query_indices)
    else:
        query_data = np.load(args.query_path)
        query_pca, query_gc = query_data["pca"], query_data["gc"]
    for phs in args.exclude_phs:
        phs = phs.strip().lower()
        if len(phs) > 0:
            if not phs.startswith("phs"):
                phs = "phs" + phs
            exclude_indices.extend(demographic_df.index[demographic_df["PHS"].apply(lambda x: x.lower() == phs)].tolist())
    exclude_indices = sorted(list(set(exclude_indices)))
    match_indices, quality_score = match_query(query_pca, query_gc, glad_pca, pruned_ds, n_matches=args.n, exclude_indices=exclude_indices, metric="minkowski", weights=explained_variance_ratio)
    if args.test_phs is None and args.test_cohort is None and args.test_ids is None:
        demographic_df = demographic_df.iloc[match_indices]
        ancestry_counts, population_codes = get_ancestry_counts(args.local_ancestry_path, args.local_ancestry_references_path, ds, demographic_df["Sample ID"].to_numpy().tolist())
        write_vcf(match_indices, quality_score, ds, demographic_df, ancestry_counts, population_codes, args)
    else:
        query_genotypes = ds.call_genotype[:, query_indices].sum("ploidy").compute(num_workers=NUM_WORKERS).to_numpy()
        query_genotype_counts = np.zeros((ds.call_genotype.sizes["variants"], 3), dtype=int)
        for i in range(0, 3):
            query_genotype_counts[:, i] = (query_genotypes == i).sum(1)
        genotypes = ds.call_genotype[:, match_indices].sum("ploidy").compute(num_workers=NUM_WORKERS).to_numpy()
        genotype_counts = np.zeros((ds.call_genotype.sizes["variants"], 3), dtype=int)
        for i in range(0, 3):
            genotype_counts[:, i] = (genotypes == i).sum(1)
        quality_score = evaluate_match(query_genotype_counts, genotype_counts)
        print("Actual lambda: {}".format(quality_score))
    return quality_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", help="path to config file containing paths to data resources", type=str, default="config.yaml")
    parser.add_argument("--query-path", help="path to query npy file", type=str, default="query.npz")
    parser.add_argument("--result-path", help="path to matching output", type=str, default="match.vcf")
    parser.add_argument("-n", help="number of controls to match for", type=int, default=-1)
    parser.add_argument("--self-described", help="only include self-described hispanics", action="store_true")
    parser.add_argument("--exclude-phs", help="phs numbers to exclude (enclose in quotations and comma-separate if more than 1)", type=str, default=None)
    parser.add_argument("--test-phs", help="treat cohort associated with this phs number(s) as query (ignores query-path, result path, and replaces exclude-phs)", type=str, default=None)
    parser.add_argument("--test-cohort", help="treat cohort(s) as query (ignores query-path, result path, and replaces exclude-phs)", type=str, default=None)
    parser.add_argument("--test-ids", help="path to file of newline separated ids that will be treated as query (ignores query-path, result path, and replaces exclude-phs)", type=str, default=None)
    parser.add_argument("--kgp", help="only search 1000 genome project (overrides exclude-phs)", action="store_true")
    args = parser.parse_args()
    
    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    # path to pca embedding
    args.pca_path = config["paths"]["pca"]
    # path to explained variance ratio for pca dimensions
    args.explained_variance_ratio_path = config["paths"]["explained_variance_ratio"]
    # path to zarr directory for genotype and other data
    args.zarr_path = config["paths"]["zarr"]
    # path to zarr directory for pruned genotype and other data
    args.pruned_zarr_path = config["paths"]["pruned_zarr"]
    # path to demographic data
    args.demography_path = config["paths"]["demography"]
    # path to msp local ancestry file
    args.local_ancestry_path = config["paths"]["local_ancestry"]
    # path to local ancestry references and population labels
    args.local_ancestry_references_path = config["paths"]["local_ancestry_references"]

    quality_score = match(args)
    print(quality_score)
