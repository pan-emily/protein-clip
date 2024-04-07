from pathlib import Path
import requests
import subprocess
from Bio import SeqIO
import random
from torch.utils.data import Dataset

from Bio import PDB
from Bio.PDB import PDBList, PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rcsbsearchapi.search import TextQuery
from rcsbsearchapi import rcsb_attributes as attrs

from rcsbsearchapi.const import CHEMICAL_ATTRIBUTE_SEARCH_SERVICE, STRUCTURE_ATTRIBUTE_SEARCH_SERVICE
from rcsbsearchapi.search import AttributeQuery
from modules import visualizations
from pathlib import Path
import requests
import subprocess
from Bio import SeqIO
import random
from torch.utils.data import Dataset
from scipy.spatial import KDTree

from Bio import PDB
from Bio.PDB import PDBList, PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import gemmi
from rcsbsearchapi.search import TextQuery
from rcsbsearchapi import rcsb_attributes as attrs

from rcsbsearchapi.const import CHEMICAL_ATTRIBUTE_SEARCH_SERVICE, STRUCTURE_ATTRIBUTE_SEARCH_SERVICE
from rcsbsearchapi.search import AttributeQuery

standard_aa_codes = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

def is_protein(chain: gemmi.Chain) -> bool:
    return any(
        res.name in standard_aa_codes for res in chain.get_polymer()
    )

def model_to_index_list(model: gemmi.Model):
    coords = []
    meta = []
    for chain in model:
        for residue in chain:
            for atom in residue:
                coords.append(list(atom.pos))
                meta.append((chain.name, residue.seqid.num, residue.name, atom.element.name, atom.name))
    assert len(coords) == sum(len(res) for chain in model for res in chain)
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
    protein1s, protein2s = _get_or_download_data()
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

def _get_or_download_data(max_sequence_length=2000, threshold=1.2):
    """
    Download pdbs and extract the binding sites. 
    """
    data_dir = Path('data')
    protein1_file_path = data_dir / 'protein1.fasta'
    protein2_file_path = data_dir / 'protein2.fasta'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if protein sequence files exist, if not, download and process
    # if not protein1_file_path.exists() or not protein2_file_path.exists():
    if True:
        warnings.simplefilter('ignore', PDBConstructionWarning)
        query = AttributeQuery("rcsb_assembly_info.polymer_entity_instance_count_protein", "equals", 2,
                    STRUCTURE_ATTRIBUTE_SEARCH_SERVICE # this constant specifies "text" service
                    )
        results = query.exec("entry")

        pdb_ids = []
        for i, assemblyid in enumerate(results):
            pdb_ids.append(assemblyid)
        

        pdbl = PDBList()
        parser = PDBParser()

        sequences_1 = {}
        sequences_2 = {}

        # pdb_ids = pdb_ids[:100]

        for pdb_id in pdb_ids:
            pdb_files_path = data_dir / 'pdb_files'
            pdbl.retrieve_pdb_file(pdb_id.lower(), pdir=pdb_files_path, file_format='pdb')
            pdb_path = f"{pdb_files_path}/pdb{pdb_id.lower()}.ent"
            if Path(pdb_path).exists():
                structure = gemmi.read_structure(str(pdb_path))
                structure.remove_waters()
                
                model = structure[0]

                if len(model) != 2:
                    continue

                coords, meta = model_to_index_list(model)
                tree = KDTree(coords)
                pairs = tree.query_pairs(r=threshold) # double check angstroms

                labels = [0] * len(coords)
                for m, (ix1, ix2) in enumerate(pairs):
                    # don't count "self" binding
                    if meta[ix1] == meta[ix2]:
                        continue
                    labels[ix1] = 1
                    labels[ix2] = 1


                protein_chains = [c.name for c in model if is_protein(c)]
                
                if len(protein_chains) == 2:
                    protein1 = "".join([meta[i][2] for i in range(len(coords)) if (meta[i][0] == protein_chains[0]) and (meta[i][2] in standard_aa_codes) and (labels[i])])
                    protein2 = "".join([meta[i][2] for i in range(len(coords)) if (meta[i][0] == protein_chains[1]) and (meta[i][2] in standard_aa_codes) and (labels[i])])

                if protein1 and protein2 and len(protein1) <= max_sequence_length and len(protein2) <= max_sequence_length:
                    sequences_1[pdb_id] = protein1
                    sequences_2[pdb_id] = protein2
                


        # Write sequences of first chain to a single FASTA file
        with open(protein1_file_path, 'w') as fasta_file_A:
            for pdb_id, sequence in sequences_1.items():
                fasta_file_A.write(f">{pdb_id}_chain_A\n{sequence}\n")

        # Write sequences of second chain to a single FASTA file
        with open(protein2_file_path, 'w') as fasta_file_B:
            for pdb_id, sequence in sequences_2.items():
                fasta_file_B.write(f">{pdb_id}_chain_B\n{sequence}\n")

    protein1s, protein2s = [], []

    with open(protein1_file_path, 'r') as f:
        for line in f.readlines():
            if not line.startswith('>'):
                protein1s.append(line.strip())
    with open(protein2_file_path, 'r') as f:
        for line in f.readlines():
            if not line.startswith('>'):
                protein2s.append(line.strip())

    assert len(protein1s) == len(protein2s), "The number of protein1s and protein2s must be the same"
    print(f"Imported {len(protein1s)} protein1s and {len(protein2s)} protein2s.")

    return protein1s, protein2s

'''
def _get_or_download_data(max_sequence_length=2000):
    """
    Get or download protein sequence data.
    
    Args:
        max_sequence_length (int, optional): Maximum sequence length for filtering sequences. Default is 2000.
        
    Returns:
        tuple: A tuple containing two lists of protein sequences (protein1s, protein2s).
    """
    data_dir = Path('data')
    protein1_file_path = data_dir / 'protein1.fasta'
    protein2_file_path = data_dir / 'protein2.fasta'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if protein sequence files exist, if not, download and process
    if not protein1_file_path.exists() or not protein2_file_path.exists():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        query = AttributeQuery("rcsb_assembly_info.polymer_entity_instance_count_protein", "equals", 2,
                    STRUCTURE_ATTRIBUTE_SEARCH_SERVICE # this constant specifies "text" service
                    )
        results = query.exec("entry")

        pdb_ids = []
        for i, assemblyid in enumerate(results):
            pdb_ids.append(assemblyid)

        pdbl = PDBList()
        parser = PDBParser()

        sequences_1 = {}
        sequences_2 = {}

        for pdb_id in pdb_ids:
            pdb_files_path = data_dir / 'pdb_files'
            pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_files_path, file_format='pdb')
            pdb_path = f"{pdb_files_path}/pdb{pdb_id.lower()}.ent"
            if Path(pdb_path).exists():
                structure = parser.get_structure(pdb_id, pdb_path)

                for model in structure:
                    chains = [chain for chain in model]
                    if len(chains) == 2:
                        sequence1 = ''.join([residue.get_resname() for residue in chains[0].get_residues() if residue.id[0] == ' '])
                        sequence2 = ''.join([residue.get_resname() for residue in chains[1].get_residues() if residue.id[0] == ' '])
                        if len(sequence1) <= max_sequence_length and len(sequence2) <= max_sequence_length:
                            sequences_1[pdb_id] = sequence1
                            sequences_2[pdb_id] = sequence2

        # Write sequences of first chain to a single FASTA file
        with open(protein1_file_path, 'w') as fasta_file_A:
            for pdb_id, sequence in sequences_1.items():
                fasta_file_A.write(f">{pdb_id}_chain_A\n{sequence}\n")

        # Write sequences of second chain to a single FASTA file
        with open(protein2_file_path, 'w') as fasta_file_B:
            for pdb_id, sequence in sequences_2.items():
                fasta_file_B.write(f">{pdb_id}_chain_B\n{sequence}\n")

    protein1s, protein2s = [], []

    with open(protein1_file_path, 'r') as f:
        for line in f.readlines():
            if not line.startswith('>'):
                protein1s.append(line.strip())
    with open(protein2_file_path, 'r') as f:
        for line in f.readlines():
            if not line.startswith('>'):
                protein2s.append(line.strip())

    assert len(protein1s) == len(protein2s), "The number of protein1s and protein2s must be the same"
    print(f"Imported {len(protein1s)} protein1s and {len(protein2s)} protein2s.")

    return protein1s, protein2s
'''

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

    if not clustered_file_path.exists():
        commands = [
            "mmseqs createdb protein2.fasta protein2DB",
            "mmseqs cluster protein2DB protein2DB_clustered tmp --min-seq-id 0.5",
            "mmseqs createtsv protein2DB protein2DB protein2DB_clustered protein2DB_clustered.tsv"
        ]

        for cmd in commands:
            _run_command(cmd)

    id_to_seq = {}
    protein2s_parsed = list(SeqIO.parse(data_dir / 'protein2.fasta', 'fasta'))
    for protein2_parsed in protein2s_parsed:
        id_to_seq[protein2_parsed.id] = str(protein2_parsed.seq)
    protein2_to_protein1 = dict(zip(protein2s, protein1s))

    clusters = {}
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



    
