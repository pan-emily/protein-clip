from pathlib import Path
import requests
import subprocess
from Bio import SeqIO
import random
from torch.utils.data import Dataset


class PeptideReceptorDataset(Dataset):
    """
    Custom PyTorch Dataset class for peptide-receptor interactions.
    
    Args:
        clusters (dict): A dictionary containing peptide-receptor interaction clusters.
        cluster_ids (list): List of cluster IDs.

    Attributes:
        clusters (dict): A dictionary containing peptide-receptor interaction clusters.
        cluster_ids (list): List of cluster IDs.
    """
    def __init__(self, clusters, cluster_ids):
        self.clusters = clusters
        self.cluster_ids = cluster_ids

    def __len__(self):
        """
        Get the number of clusters in the dataset.
        
        Returns:
            int: Number of clusters.
        """
        return len(self.cluster_ids)

    def __getitem__(self, idx):
        """
        Get a pair of peptide and receptor sequences from a cluster.
        
        Args:
            idx (int): Index of the cluster.
            
        Returns:
            tuple: A tuple containing two sequences (peptide_sequence, receptor_sequence).
        """
        curr_cluster = self.clusters[self.cluster_ids[idx]]
        curr_pair = random.choice(curr_cluster)
        peptide_sequence = curr_pair[0]
        receptor_sequence = curr_pair[1]
        return peptide_sequence, receptor_sequence

def generate_datasets():
    """
    Generate train, validation, and test datasets for peptide-receptor interactions.
    
    Returns:
        tuple: A tuple containing train, validation, and test datasets.
    """
    peptides, receptors = _get_or_download_data()
    clusters = _cluster_data(peptides, receptors)
    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids) 

    num_train = int(0.7 * len(cluster_ids))
    num_val = int(0.15 * len(cluster_ids))

    train_clusters = cluster_ids[:num_train]
    val_clusters = cluster_ids[num_train:num_train+num_val]
    test_clusters = cluster_ids[num_train+num_val:]

    train_dataset = PeptideReceptorDataset(clusters, train_clusters)
    val_dataset = PeptideReceptorDataset(clusters, val_clusters)
    test_dataset = PeptideReceptorDataset(clusters, test_clusters)
    
    return train_dataset, val_dataset, test_dataset 

def _get_or_download_data():
    """
    Get or download peptide and receptor sequence data.
    
    Returns:
        tuple: A tuple containing two lists of sequences (peptides, receptors).
    """
    data_dir = Path('data')
    peptide_file_path = data_dir / 'peptide.fasta'
    receptor_file_path = data_dir / 'receptor.fasta'
    data_dir.mkdir(parents=True, exist_ok=True)

    peptide_url = 'http://bioinfo.dcc.ufmg.br/propedia/public/download/peptide.fasta'
    receptor_url = 'http://bioinfo.dcc.ufmg.br/propedia/public/download/receptor.fasta'
    if not peptide_file_path.exists():
        peptide_data = requests.get(peptide_url).text
        with open(peptide_file_path, 'w') as f:
            f.write(peptide_data)
    if not receptor_file_path.exists():
        receptor_data = requests.get(receptor_url).text
        with open(receptor_file_path, 'w') as f:
            f.write(receptor_data)

    peptides, receptors = [], []
    with open(peptide_file_path, 'r') as f:
        for line in f.readlines():
            if not line.startswith('>'):
                peptides.append(line.strip())
    with open(receptor_file_path, 'r') as f:
        for line in f.readlines():
            if not line.startswith('>'):
                receptors.append(line.strip())

    assert len(peptides) == len(receptors), "The number of peptides and receptors must be the same"
    print(f"Imported {len(peptides)} peptides and {len(receptors)} receptors.")
    return peptides, receptors

def _cluster_data(peptides, receptors):
    """
    Cluster peptide-receptor interaction data based on sequence similarity.
    
    Args:
        peptides (list): List of peptide sequences.
        receptors (list): List of receptor sequences.
        
    Returns:
        dict: A dictionary containing peptide-receptor interaction clusters.
    """
    data_dir = Path('data')
    clustered_file_path = data_dir / 'receptorDB_clustered.tsv'

    if not clustered_file_path.exists():
        commands = [
            "mmseqs createdb receptor.fasta receptorDB",
            "mmseqs cluster receptorDB receptorDB_clustered tmp --min-seq-id 0.5",
            "mmseqs createtsv receptorDB receptorDB receptorDB_clustered receptorDB_clustered.tsv"
        ]

        for cmd in commands:
            _run_command(cmd)

    id_to_seq = {}
    receptors_parsed = list(SeqIO.parse(data_dir / 'receptor.fasta', 'fasta'))
    for receptor_parsed in receptors_parsed:
        id_to_seq[receptor_parsed.id] = str(receptor_parsed.seq)
    receptor_to_peptide = dict(zip(receptors, peptides))

    clusters = {}
    with open(clustered_file_path, 'r') as f:
        for line in f:
            cluster_id, receptor_id = line.strip().split("\t")
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            receptor_sequence = id_to_seq[receptor_id]
            peptide_sequence = receptor_to_peptide[receptor_sequence]
            clusters[cluster_id].append((peptide_sequence, receptor_sequence))

    return clusters

def _run_command(command):
    """
    Run a shell command.
    
    Args:
        command (str): Shell command to be executed.
    """
    print(f"Running command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=Path('data'))
        print(f"Command output: {result.stdout}\n\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")


    
