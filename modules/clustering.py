from Bio import SeqIO
import random
import torch 
import numpy as np 

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    return seed_value

def cluster(receptors, peptides, seed):
    id_to_seq = {}
    receptors_parsed = list(SeqIO.parse('receptor.fasta', 'fasta'))
    for receptor_parsed in receptors_parsed:
        id_to_seq[receptor_parsed.id] = str(receptor_parsed.seq)
    receptor_to_peptide = dict(zip(receptors, peptides))

    with open('receptorDB_clu.tsv', 'r') as f:
        clusters = {}
        for line in f:
            cluster_id, receptor_id = line.strip().split("\t")
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            receptor_sequence = id_to_seq[receptor_id]
            peptide_sequence = receptor_to_peptide[receptor_sequence]
            clusters[cluster_id].append((peptide_sequence, receptor_sequence))

    set_seed(seed)

    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids)
    num_train = int(0.7 * len(cluster_ids))
    num_val = int(0.15 * len(cluster_ids))
    train_clusters = cluster_ids[:num_train]
    val_clusters = cluster_ids[num_train:num_train+num_val]
    test_clusters = cluster_ids[num_train+num_val:]

    print(f"Clustering complete")

    return clusters, train_clusters, val_clusters, test_clusters 