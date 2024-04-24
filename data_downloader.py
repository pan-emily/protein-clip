import torch
import json
from Bio import PDB
from pathlib import Path
import numpy as np
from ppiref.utils.misc import download_from_zenodo
from Bio.PDB.Polypeptide import protein_letters_3to1

def load_or_process_data(ppiref_dir="/home/ubuntu/miniconda3/envs/protein-clip/lib/python3.10/site-packages/ppiref/", contact_name='6A'):
    json_path = Path(ppiref_dir) / f"data/splits/ppiref_{contact_name}_filtered_clustered_04.json"
    data_dir = Path('data')
    protein1_path = data_dir / 'protein1.pt'
    protein2_path = data_dir / 'protein2.pt'
    data_dir.mkdir(parents=True, exist_ok=True)

    if True or not protein1_path.exists() or not protein2_path.exists():
        pdb_ids = load_pdb_ids(json_path)
        process_and_save_pdb_data(pdb_ids, protein1_path, protein2_path, ppiref_dir, contact_name)

    return torch.load(protein1_path), torch.load(protein2_path)

def load_pdb_ids(json_path):
    with json_path.open('r') as file:
        return json.load(file)['folds']['whole']

def process_and_save_pdb_data(pdb_ids, protein1_path, protein2_path, ppiref_dir, contact_name):
    file = download_from_zenodo(f'ppi_{contact_name}.zip')
    pdbl = PDB.PDBList()
    parser = PDB.PDBParser()
    processed_data_protein1, processed_data_protein2 = [], []

    for pdb_id in pdb_ids:#[:1000]:
        pdb_path = Path(ppiref_dir) / f'data/ppiref/ppi_{contact_name}/{pdb_id[1:3]}/{pdb_id}.pdb'
        if pdb_path.exists():
            chains = parser.get_structure("PDB_structure", pdb_path)[0]
            if len(chains) == 2:
                chain1 = True
                for chain in chains:
                    feats, coors = convert_chain_to_graph(chain, 1000)
                    if chain1:
                        processed_data_protein1.append((feats, coors))
                    else:
                        processed_data_protein2.append((feats, coors))
                    chain1 = False

    torch.save(processed_data_protein1, protein1_path)
    torch.save(processed_data_protein2, protein2_path)

def convert_chain_to_graph(chain, max_residues):
    feats, coors = [], []
    count = 0
    for residue in chain.get_residues():
        if count >= max_residues:
            break
        if PDB.Polypeptide.is_aa(residue, standard=True):  # Check if residue is a standard amino acid
            try:
                feats.append(ord(protein_letters_3to1[residue.resname]))
            except KeyError:
                feats.append(ord('X'))  # Use 'X' for unknown or non-standard amino acids
        residue_coords = np.array([atom.get_coord() for atom in residue if atom.element != 'H'])  # Exclude hydrogen atoms
        if residue_coords.size > 0:
            coors.append(np.mean(residue_coords, axis=0))
        count += 1
    pad_size = max_residues - count
    if pad_size > 0:
        pad_feats = np.full(pad_size, ord(' '))
        pad_coors = np.zeros((pad_size, 3))
        feats = np.concatenate((feats, pad_feats), axis=0)
        coors = np.concatenate((coors, pad_coors), axis=0)
    return np.array(feats), np.array(coors, dtype=np.float32)  # Return sequence string and coordinates as numpy array

load_or_process_data()