from pathlib import Path
import requests
import subprocess
from Bio import SeqIO
import torch
import random
from torch.utils.data import Dataset

from rcsbsearchapi.const import CHEMICAL_ATTRIBUTE_SEARCH_SERVICE, STRUCTURE_ATTRIBUTE_SEARCH_SERVICE
from rcsbsearchapi.search import AttributeQuery
from modules import visualizations
from scipy.spatial import KDTree

from Bio import PDB
from Bio.PDB import PDBList, PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import gemmi
from rcsbsearchapi.search import TextQuery
from rcsbsearchapi import rcsb_attributes as attrs
import gzip
from io import BytesIO
import os
import json
import numpy as np

standard_aa_codes = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

def extract_residue_features(residue):
    # one hot encode amino acid type
    aa_types = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_onehot = [1.0 if residue.get_resname() == aa else 0.0 for aa in aa_types]
    return aa_onehot

def is_protein(chain: gemmi.Chain) -> bool:
    return any(
        res.name in standard_aa_codes for res in chain.get_polymer()
    )

def chain_to_index_list(chain: gemmi.Chain):
    coords = []
    meta = []
    for residue in chain:
        for atom in residue:
            coords.append(list(atom.pos))
            meta.append((chain.name, residue.seqid.num, residue.name, atom.element.name, atom.name))
    assert len(coords) == sum(len(res) for res in chain)
    return coords, meta

class ProteinProteinDataset(Dataset):
    """
    Custom PyTorch Dataset class for protein-protein interactions.
    
    Args:
        clusters (dict): A dictionary containing protein interaction clusters.
        cluster_ids (list): List of cluster IDs.

    Attributes:
        clusters (dict): A dictionary containing protein interaction clusters.
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
        Get a pair of protein sequences from a cluster.
        
        Args:
            idx (int): Index of the cluster.
            
        Returns:
            tuple: A tuple containing two protein sequences (protein1_sequence, protein2_sequence).
        """
        curr_cluster = self.clusters[self.cluster_ids[idx]]
        if curr_cluster:
            curr_pair = random.choice(curr_cluster)
            protein1_sequence = curr_pair[0]
            protein2_sequence = curr_pair[1]
            return protein1_sequence, protein2_sequence
        # Return empty strings if no sequences in the cluster
        return '', ''

def generate_datasets():
    """
    Generate train, validation, and test datasets for protein-protein interactions.
    
    Returns:
        tuple: A tuple containing train, validation, and test datasets.
    """
    protein1s, protein2s = _from_ppiref_get_or_download_data()
    # TODO: Redo clustering logic
    clusters = _cluster_data(protein1s, protein2s)
    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids) 

    num_train = int(0.7 * len(cluster_ids))
    num_val = int(0.15 * len(cluster_ids))

    train_clusters = cluster_ids[:num_train]
    val_clusters = cluster_ids[num_train:num_train+num_val]
    test_clusters = cluster_ids[num_train+num_val:]

    train_dataset = ProteinProteinDataset(clusters, train_clusters)
    val_dataset = ProteinProteinDataset(clusters, val_clusters)
    test_dataset = ProteinProteinDataset(clusters, test_clusters)
    
    return train_dataset, val_dataset, test_dataset 

def generate_unpaired_datasets(max_residues=1000):
    """
    Generate train, validation, and test datasets for protein-protein interactions.
    
    Returns:
        tuple: A tuple containing train, validation, and test datasets.
    """
    protein1s, protein2s = _from_ppiref_get_or_download_data(max_residues=max_residues)
    print(len(protein1s), len(protein2s))
    # turn these two lists into one singular list
    all_proteins = protein1s + protein2s

    # shuffle all_proteins
    random.shuffle(all_proteins)

    num_train = int(0.7 * len(all_proteins))
    num_val = int(0.15 * len(all_proteins))

    train_data = all_proteins[:num_train]
    val_data = all_proteins[num_train:num_train+num_val]
    test_data = all_proteins[num_train+num_val:]

    tensor_datasets = []
    for dataset in [train_data, val_data, test_data]:
        feats_list = [item[0].clone().detach() for item in dataset]
        feats_tensor = torch.stack(feats_list).long()  # Convert to long only if necessary
        coors_list = [item[1].clone().detach() for item in dataset]
        coors_tensor = torch.stack(coors_list).double()  # Convert to double only if necessary

        tensor_dataset = TensorDataset(feats_tensor, coors_tensor)
        tensor_datasets.append(tensor_dataset)

    print(f'Data loaded. Number of samples: {len(all_proteins)}')
    
    return tensor_datasets

def convert_chain_to_graph(chain, max_residues=1000):
    feats = []
    coors = []

    residues = list(chain.get_residues())
    num_residues = len(residues)

    for i, residue in enumerate(residues):
        if i >= max_residues:
            break

        # extract residue features
        residue_feats = extract_residue_features(residue)
        # residue_feats = PDB.Polypeptide.protein_letters_3to1.get(residue.resname.upper(), 'X')
        feats.append(residue_feats)

        # residue coords
        residue_coords = [atom.get_coord() for atom in residue.get_atoms()]
        center_of_mass = np.mean(residue_coords, axis=0)
        coors.append(center_of_mass)

    # Pad or truncate features and coordinates
    pad_size = max_residues - num_residues
    if pad_size > 0:
        pad_feats = np.zeros((pad_size, len(feats[0])))
        pad_coors = np.zeros((pad_size, 3))
        feats = np.concatenate((feats, pad_feats), axis=0)
        coors = np.concatenate((coors, pad_coors), axis=0)
    else:
        feats = feats[:max_residues]
        coors = coors[:max_residues]

    # Create adjacency matrix based on sequence connectivity
    """edges = np.zeros((max_residues, max_residues))
    for i in range(min(num_residues, max_residues) - 1):
        edges[i, i + 1] = 1
        edges[i + 1, i] = 1"""

    # Convert to PyTorch tensors
    feats = torch.tensor(feats, dtype=torch.float32)
    coors = torch.tensor(coors, dtype=torch.float32)
    # edges = torch.tensor(edges, dtype=torch.float32)

    return feats, coors, # edges

def _from_ppiref_get_or_download_data(max_sequence_length=2000, topK=50, gcloud=True, ppiref_dir = "/home/ubuntu/miniconda3/envs/protein-clip/lib/python3.10/site-packages/ppiref/", contact_name='6A', max_residues=1000):
    """
    Download pdbs and extract the binding sites. 
    """
    from ppiref.extraction import PPIExtractor
    from ppiref.definitions import PPIREF_TEST_DATA_DIR
    from ppiref.utils.misc import download_from_zenodo
    file = download_from_zenodo(f'ppi_{contact_name}.zip')
    ppi_master_file_name = f'ppiref_{contact_name}_filtered_clustered_04.json'

    data_dir = Path('data')
    protein1_file_path = data_dir / 'protein1.pt'
    protein2_file_path = data_dir / 'protein2.pt'
    if not protein1_file_path.exists() or not protein2_file_path.exists():
        # get pdb ids (file names) from /home/ubuntu/miniconda3/envs/protein-clip/lib/python3.10/site-packages/ppiref/data/splits/ppiref_6A_filtered_clustered_04.json
        
        json_path = ppiref_dir + "data/splits/" + ppi_master_file_name
        with open(json_path, 'r') as json_file:
            results = json.load(json_file)
        pdb_ids = results['folds']['whole']
        print(f'{ppi_master_file_name} has {len(pdb_ids)} binding pairs.')

        pdbl = PDBList()
        parser = PDBParser()

        processed_data_protein1 = []; processed_data_protein2 = []
        
        pdb_files_path_ppiref = ppiref_dir + 'data/ppiref/ppi_' + contact_name
        for pdb_id in pdb_ids:
            pdb_path = f"{pdb_files_path_ppiref}/{pdb_id[1:3]}/{pdb_id}.pdb"
            if Path(pdb_path).exists():
                # testing using primary structure; get sequences from each chain separately
                structure = parser.get_structure("PDB_structure", pdb_path)
                sequences = {}; graphs = {}
                model = structure[0]
                chain_ids = [x.lower() for x in pdb_id.split('_')[1:]]
                for chain in model:
                    curr_chain_id = chain.id.lower()
                    if curr_chain_id in chain_ids:
                        feats, coors = convert_chain_to_graph(chain, max_residues)
                        graphs[curr_chain_id] = (feats, coors)
                        seq = []
                        for residue in chain:
                            if PDB.is_aa(residue, standard=True):
                                seq.append(PDB.Polypeptide.protein_letters_3to1.get(residue.resname.upper(), 'X'))
                        sequences[curr_chain_id] = ''.join(seq)
            if len(sequences) != 2: f"Expected 2 chains, got {len(sequences)} chains"
            else:
                # Write graphs of first chain to a single pt file
                processed_data_protein1.append(graphs[chain_ids[0]])
                processed_data_protein2.append(graphs[chain_ids[1]])
        torch.save(processed_data_protein1, protein1_file_path)
        torch.save(processed_data_protein2, protein2_file_path)
    else:
        print(f"Loading existing data from {protein1_file_path} and {protein2_file_path}")
    protein1s = torch.load(protein1_file_path)
    protein2s = torch.load(protein2_file_path)

    assert len(protein1s) == len(protein2s), "The number of protein1s and protein2s must be the same"
    print(f"Imported {len(protein1s)} protein1s and {len(protein2s)} protein2s.")

    return protein1s, protein2s

def _cluster_data(protein1s, protein2s):
    """
    Cluster protein data based on sequence similarity.
    
    Args:
        protein1s (list): List of protein sequences (chain A).
        protein2s (list): List of protein sequences (chain B).
        
    Returns:
        dict: A dictionary containing protein interaction clusters.
    """
    data_dir = Path('data')
    clustered_file_path = data_dir / 'protein2DB_clustered.tsv'

    id_to_seq = {}
    for i, e in enumerate(zip(protein2s, protein1s)):
        id_to_seq[i] = e
    protein2_to_protein1 = dict(zip(protein2s, protein1s))

    clusters = {}
    '''
    with open(clustered_file_path, 'r') as f:
        for line in f:
            cluster_id, protein2_id = line.strip().split("\t")
            if cluster_id not in clusters:
                clusters[cluster_id] = []

            protein2_sequence = id_to_seq[protein2_id]
            if protein2_sequence in protein2_to_protein1:
                protein1_sequence = protein2_to_protein1[protein2_sequence]
                clusters[cluster_id].append((protein1_sequence, protein2_sequence))
            else:
                print(f"Missing sequence match for: {protein2_sequence}")
    '''
    clusters = id_to_seq
    print(len(clusters))
    clusters = {cid: clust for cid, clust in clusters.items() if clust}
    print(len(clusters))
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



    
