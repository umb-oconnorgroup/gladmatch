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
            keep = np.concatenate([np.arange(len(match_indices) - distances.shape[0]), len(match_indices) - distances.shape[0] + np.random.permutation(n_keep_from_latest)])
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
    distances = cdist(query_embedding, embedding, metric=metric, **kwargs)
    if exclude_indices is not None and len(exclude_indices) > 0:
        distances[:, exclude_indices] = 1e10
    if n_matches == -1:
        n_matches = query_embedding.shape[0]
    match_indices = seed_match(np.copy(distances), n_matches)
    genotypes = dataset.call_genotype[:, match_indices].sum("ploidy").compute().to_numpy()
    genotype_counts = np.zeros((dataset.call_genotype.sizes["variants"], 3), dtype=int)
    for i in range(0, 3):
        genotype_counts[:, i] = (genotypes == i).sum(1)
    quality_score = evaluate_match(query_genotype_counts, genotype_counts)
    return match_indices, quality_score, genotype_counts

def build_demographic_data(demography_path, ds):
    columns = ["Cohort", "PHS", "Country", "State", "City", "A", "B", "C", "Sex", "SelfDescribedStatus", "HispanicJustification"]
    demographic_df = pd.read_csv(demography_path, header=None, index_col=0, sep='\t', names=columns)
    demographic_df = demographic_df[~demographic_df.index.duplicated(keep="first")]
    sample_ids = ds.sample_id.compute().to_numpy().tolist()
    demographic_sample_ids = demographic_df.index.to_numpy().tolist()
    demographic_df.drop(set(demographic_sample_ids).difference(set(sample_ids)), inplace=True)
    missing_sample_ids = list(set(sample_ids).difference(set(demographic_sample_ids)))
    if len(missing_sample_ids) > 0:
        demographic_df = demographic_df.append(pd.DataFrame(np.array(['Unknown'], dtype=object).repeat(len(missing_sample_ids)).reshape(-1, 1).repeat(len(demographic_df.columns), axis=1), index=missing_sample_ids, columns=demographic_df.columns))
    demographic_df.fillna('Unknown', inplace=True)
    demographic_df = demographic_df.reindex(sample_ids)
    demographic_df["Sample ID"] = demographic_df.index
    demographic_df.reset_index(drop=True, inplace=True)
    return demographic_df

def get_ancestry_counts(local_ancestry_path, ds, match_ids):
    positions = ds.variant_position.compute().to_numpy()
    variant_contigs = ds.variant_contig.compute().to_numpy()
    haploid_match_ids = [match_id + ".{}".format(i) for match_id in match_ids for i in [0, 1]]
    with open(local_ancestry_path, "r") as f:
        populations = f.readline().strip("\n").split(": ", 1)[1].split("\t")
    population_codes = [tuple(population_entry.split("=")) for population_entry in populations]
    population_codes = {int(code): population for population, code in population_codes}
    ancestry_counts = np.zeros((ds.call_genotype.sizes["variants"], len(population_codes)), dtype=int)
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

def write_vcf(match_indices, quality_score, genotype_counts, dataset, demographic_df, ancestry_counts, population_codes, args):
    genotypes = dataset.call_genotype
    allele_counts = genotypes[:, match_indices].sum(["ploidy", "samples"]).compute().to_numpy()
    allele_frequencies = allele_counts / (len(match_indices) * 2)
    chromosomes = [dataset.contigs[contig_index] for contig_index in dataset.variant_contig.compute().to_numpy()]
    positions = dataset.variant_position.compute().to_numpy()
    alleles = map(lambda x: [x[0].decode(), x[1].decode()], dataset.variant_allele.compute().to_numpy())
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
        if args.exclude_phs is not None:
            f.write("##match-parameter:exclude-phs={}\n".format(args.exclude_phs))
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

def match(args):
    glad_pca = np.load(args.pca_path)
    explained_variance_ratio = np.load(args.explained_variance_ratio_path)
    query_data = np.load(args.query_path)
    query_pca, query_gc = query_data["pca"], query_data["gc"]
    ds = sg.load_dataset(args.zarr_path)
    pruned_ds = sg.load_dataset(args.pruned_zarr_path)
    demographic_df = build_demographic_data(args.demography_path, ds)
    exclude_indices = []
    if args.self_described:
        exclude_indices.extend(demographic_df.index[demographic_df["SelfDescribedStatus"] != "Hispanic"].tolist())
    if args.exclude_phs is not None:
        for phs in args.exclude_phs.split(","):
            phs = phs.strip().lower()
            if len(phs) > 0:
                if not phs.startswith("phs"):
                    phs = "phs" + phs
                exclude_indices.extend(demographic_df.index[demographic_df["PHS"].apply(lambda x: x.lower() == phs)].tolist())
    exclude_indices = sorted(list(set(exclude_indices)))
    match_indices, quality_score, genotype_counts = match_query(query_pca, query_gc, glad_pca, pruned_ds, n_matches=args.n, exclude_indices=exclude_indices, metric="minkowski", weights=explained_variance_ratio)
    demographic_df = demographic_df.iloc[match_indices]
    ancestry_counts, population_codes = get_ancestry_counts(args.local_ancestry_path, ds, demographic_df["Sample ID"].to_numpy().tolist())
    write_vcf(match_indices, quality_score, genotype_counts, ds, demographic_df, ancestry_counts, population_codes, args)
    return quality_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pca_path", help="path to pca embedding", type=str)
    parser.add_argument("explained_variance_ratio_path", help="path to explained variance ratio for pca dimensions", type=str)
    parser.add_argument("zarr_path", help="path to zarr directory for genotype and other data", type=str)
    parser.add_argument("pruned_zarr_path", help="path to zarr directory for pruned genotype and other data", type=str)
    parser.add_argument("query_path", help="path to query npy file", type=str)
    parser.add_argument("demography_path", help="path to demographic data", type=str)
    parser.add_argument("local_ancestry_path", help="path to msp local ancestry file", type=str)
    parser.add_argument("result_path", help="path to matching output", type=str)
    parser.add_argument("-n", help="number of controls to match for", type=int, default=-1)
    parser.add_argument("--self-described", help="only include self-described hispanics", action="store_true")
    parser.add_argument("--exclude-phs", help="phs numbers to exclude (enclose in quotations and comma-separate if more than 1)", type=str, default=None)
    args = parser.parse_args()
    quality_score = match(args)
    print(quality_score)
