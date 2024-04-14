from pathlib import Path
import requests
import subprocess
from Bio import SeqIO
import random
from torch.utils.data import Dataset

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
import requests
import gzip
from io import BytesIO
import os
import json

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

def _from_ppiref_get_or_download_data(max_sequence_length=2000, topK=50, gcloud=True, ppiref_dir = "/home/ubuntu/miniconda3/envs/protein-clip/lib/python3.10/site-packages/ppiref/", contact_name='6A'):
    """
    Download pdbs and extract the binding sites. 
    """
    from ppiref.extraction import PPIExtractor
    from ppiref.definitions import PPIREF_TEST_DATA_DIR
    from ppiref.utils.misc import download_from_zenodo
    file = download_from_zenodo(f'ppi_{contact_name}.zip')
    ppi_master_file_name = f'ppiref_{contact_name}_filtered_clustered_04.json'

    data_dir = Path('data')
    protein1_file_path = data_dir / 'protein1.fasta'
    protein2_file_path = data_dir / 'protein2.fasta'
    if not protein1_file_path.exists() or not protein2_file_path.exists():
        # get pdb ids (file names) from /home/ubuntu/miniconda3/envs/protein-clip/lib/python3.10/site-packages/ppiref/data/splits/ppiref_6A_filtered_clustered_04.json
        json_path = ppiref_dir + "data/splits/" + ppi_master_file_name
        with open(json_path, 'r') as json_file:
            results = json.load(json_file)
        pdb_ids = results['folds']['whole']
        print(f'{ppi_master_file_name} has {len(pdb_ids)} binding pairs.')

        pdbl = PDBList()
        parser = PDBParser()

        sequences_1 = {}
        sequences_2 = {}
        
        pdb_files_path_ppiref = ppiref_dir + 'data/ppiref/ppi_' + contact_name
        for pdb_id in pdb_ids:
            pdb_path = f"{pdb_files_path_ppiref}/{pdb_id[1:3]}/{pdb_id}.pdb"
            if Path(pdb_path).exists():
                # testing using primary structure; get sequences from each chain separately
                structure = parser.get_structure("PDB_structure", pdb_path)
                sequences = {}
                model = structure[0]
                chain_ids = [x.lower() for x in pdb_id.split('_')[1:]]
                for chain in model:
                    curr_chain_id = chain.id.lower()
                    if curr_chain_id in chain_ids:
                        seq = []
                        for residue in chain:
                            if PDB.is_aa(residue, standard=True):
                                seq.append(PDB.Polypeptide.protein_letters_3to1.get(residue.resname.upper(), 'X'))
                        sequences[curr_chain_id] = ''.join(seq)
            if len(sequences) != 2: f"Expected 2 chains, got {len(sequences)} chains"
            else:
                # Write sequences of first chain to a single FASTA file
                with open(protein1_file_path, 'a') as fasta_file_A:
                    fasta_file_A.write(f">{pdb_id.split('_')[0]}_chain_{chain_ids[0]}\n{sequences[chain_ids[0]]}\n")

                # Write sequences of second chain to a single FASTA file
                with open(protein2_file_path, 'a') as fasta_file_B:
                    fasta_file_B.write(f">{pdb_id.split('_')[0]}_chain_{chain_ids[1]}\n{sequences[chain_ids[1]]}\n")
            

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

def _get_or_download_data(max_sequence_length=2000, topK=50, gcloud=True):
    """
    Download pdbs and extract the binding sites. 
    """
    data_dir = Path('data')
    protein1_file_path = data_dir / 'protein1.fasta'
    protein2_file_path = data_dir / 'protein2.fasta'
    if not protein1_file_path.exists() or not protein2_file_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

        warnings.simplefilter('ignore', PDBConstructionWarning)
        if not gcloud:
            query = AttributeQuery("rcsb_assembly_info.polymer_entity_instance_count_protein", "equals", 2,
                        STRUCTURE_ATTRIBUTE_SEARCH_SERVICE # this constant specifies "text" service
                        )
            results = query.exec("entry")
        else:
            blob_url = 'https://storage.googleapis.com/rohit-general/pdb_structures/pdb_ids.txt.gz'
            results_gz = requests.get(blob_url)

            with gzip.open(BytesIO(results_gz.content), 'rt') as f:
                results = f.read()
        
        pdb_ids = [assemblyid for assemblyid in results.split('\n')]

        pdbl = PDBList()
        parser = PDBParser()

        sequences_1 = {}
        sequences_2 = {}

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

                chain1, chain2 = model[0], model[1]
                coords1, meta1 = chain_to_index_list(chain1)
                coords2, meta2 = chain_to_index_list(chain2)

                tree1 = KDTree(coords1)
                tree2 = KDTree(coords2)

                closest_residues = []
                added_pairs = set() # keep track of pairs of indices

                # find closest residues in group 1 for each residue in group 2
                for j in range(len(coords2)):
                    dist, i = tree1.query(coords2[j], k=1)
                    # if we haven't seen this pair before
                    if (i, j) not in added_pairs:
                        added_pairs.add((i, j))
                        closest_residues.append((dist, (i, j)))

                # find closest residues in group 1 for each residue in group 2
                for i in range(len(coords1)):
                    # if we haven't seen this pair before
                    dist, j = tree2.query(coords1[i], k=1)
                    if (i, j) not in added_pairs:
                        added_pairs.add((i, j))
                        closest_residues.append((dist, (i, j)))

                # find top K
                closest_residues.sort()
                top_residues_1 = closest_residues[:topK]
                
                # now sort by position
                top_residues_1.sort(key=lambda x: x[1])
                fragments = []
                curr = []
                window_size = 10
                for _, pair in top_residues_1:
                    # check if current index of is within window size
                    if not curr or (pair[0] - curr[-1][0] <= window_size):
                        curr.append(pair)
                    else:
                        fragments.append(curr)
                        curr = []

                if curr:
                    fragments.append(curr)

                print(fragments)

                # convert to residues
                group1 = [''.join([meta1[pair[0]][2] for pair in fragment]) for fragment in fragments]
                group2 = [''.join([meta2[pair[1]][2] for pair in fragment]) for fragment in fragments]

                print(group1)

                sequences_1[pdb_id] = group1
                sequences_2[pdb_id] = group2

            # Write sequences of first chain to a single FASTA file
            with open(protein1_file_path, 'w') as fasta_file_A:
                for pdb_id, sequence in sequences_1.items():
                    fasta_file_A.write(f">{pdb_id}_chain_A\n{sequence}\n")

            # Write sequences of second chain to a single FASTA file
            with open(protein2_file_path, 'w') as fasta_file_B:
                for pdb_id, sequence in sequences_2.items():
                    fasta_file_B.write(f">{pdb_id}_chain_B\n{sequence}\n")

            try:
                os.remove(pdb_path)
            except OSError as e:
                print(f"Error deleting file: {pdb_path}")
                print(f"Error message: {str(e)}")
            

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



    
