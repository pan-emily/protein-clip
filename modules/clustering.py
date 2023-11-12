from Bio import SeqIO
import random
import torch 
import numpy as np 
import subprocess 
from torch.utils.data import Dataset

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

# Function to run a shell command and capture its output
def run_command(command):
    try:
        # This will execute the command and capture the output
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")

def filter_empty_clusters(clusters):
    # Remove any clusters that are empty
    return {cluster_id: cluster_data for cluster_id, cluster_data in clusters.items() if cluster_data}

def cluster(peptides, receptors, seed):
    # Commands to be executed
    commands = [
        "/Users/emilypan/Documents/Caltech/protein-clip/mmseqs version",
        "mmseqs createdb receptor.fasta receptorDB",
        "mmseqs cluster receptorDB receptorDB_clu tmp --min-seq-id 0.5",
        "mmseqs createtsv receptorDB receptorDB receptorDB_clu receptorDB_clu.tsv"
    ]

    # Run commands
    for cmd in commands:
        print(f"Running command: {cmd}")
        run_command(cmd)

    id_to_seq = {}
    receptors_parsed = list(SeqIO.parse('receptor.fasta', 'fasta'))
    for receptor_parsed in receptors_parsed:
        id_to_seq[receptor_parsed.id] = str(receptor_parsed.seq)
    
    # Debug: Print out receptors and peptides to ensure they match
    # print(f"Receptors: {receptors}")
    # print(f"Peptides: {peptides}")

    receptor_to_peptide = dict(zip(receptors, peptides))

    with open('receptorDB_clu.tsv', 'r') as f:
        clusters = {}
        for line in f:
            cluster_id, receptor_id = line.strip().split("\t")
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            receptor_sequence = id_to_seq[receptor_id]
            
            # Safely access the receptor_to_peptide with a default value
            peptide_sequence = receptor_to_peptide.get(receptor_sequence, None)
            
            # Debug: Check if the sequence is not found in the dictionary
            if peptide_sequence is None:
                # print(f"Sequence not found for receptor ID: {receptor_id}")
                continue  # Skip this iteration
            
            clusters[cluster_id].append((peptide_sequence, receptor_sequence))

    set_seed(seed)
    # clusters = filter_empty_clusters(clusters)

    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids)
    num_train = int(0.7 * len(cluster_ids))
    num_val = int(0.15 * len(cluster_ids))
    train_clusters = cluster_ids[:num_train]
    val_clusters = cluster_ids[num_train:num_train+num_val]
    test_clusters = cluster_ids[num_train+num_val:]

    print(f"Clustering complete {len(cluster_ids)} clusters.")

    return clusters, train_clusters, val_clusters, test_clusters 
