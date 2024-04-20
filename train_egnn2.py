import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from egnn_pytorch import EGNN_Network
from pathlib import Path
from datetime import datetime
import os, json, random
import numpy as np
from Bio import PDB
from Bio.PDB import PDBList, PDBParser

# default tensor type to float32 to use less memory
torch.set_default_dtype(torch.float32)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
base_path = Path(os.getcwd()) / 'runs' / timestamp
base_path.mkdir(parents=True, exist_ok=True)
print(f"All run info will be saved to {base_path}")

# Constants
BATCH_SIZE = 32
EPOCHS = 1000

class NewProteinDataset(Dataset):
    """ Dataset for loading protein data tuples (features, coordinates). """
    def __init__(self, protein_data):
        self.features = [self._to_tensor(row[0], dtype=torch.long) for row in protein_data]
        self.coordinates = [self._to_tensor(row[1], dtype=torch.float) for row in protein_data]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.coordinates[idx]

    @staticmethod
    def _to_tensor(data, dtype):
        return data.clone().detach().to(dtype=dtype) if torch.is_tensor(data) else torch.tensor(data, dtype=dtype)

def generate_unpaired_datasets(max_residues=1000):
    protein1s, protein2s = _from_ppiref_get_or_download_data(max_residues=max_residues)
    all_proteins = protein1s + protein2s
    random.shuffle(all_proteins)

    split_train = int(0.7 * len(all_proteins))
    split_val = int(0.15 * len(all_proteins))

    return (NewProteinDataset(all_proteins[:split_train]),
            NewProteinDataset(all_proteins[split_train:split_train+split_val]),
            NewProteinDataset(all_proteins[split_train+split_val:]))

def _from_ppiref_get_or_download_data(max_residues=1000, ppiref_dir="path/to/ppiref", contact_name='6A'):
    data_dir = Path('data')
    protein1_path = data_dir / 'protein1.pt'
    protein2_path = data_dir / 'protein2.pt'
    data_dir.mkdir(parents=True, exist_ok=True)

    if not protein1_path.exists() or not protein2_path.exists():
        download_and_process_data(protein1_path, protein2_path, ppiref_dir, contact_name)
    else:
        print(f"Loading existing data from {protein1_path} and {protein2_path}")

    return torch.load(protein1_path), torch.load(protein2_path)

def download_and_process_data(protein1_path, protein2_path, ppiref_dir, contact_name):
    json_path = Path(ppiref_dir) / "data/splits" / f'ppiref_{contact_name}_filtered_clustered_04.json'
    with open(json_path, 'r') as file:
        pdb_ids = json.load(file)['folds']['whole']

    pdbl = PDBList()
    parser = PDBParser()
    processed_data_protein1, processed_data_protein2 = [], []

    for pdb_id in pdb_ids:
        pdb_path = Path(ppiref_dir) / 'data/ppiref/ppi' / contact_name / pdb_id[1:3] / f'{pdb_id}.pdb'
        if pdb_path.exists():
            structure = parser.get_structure("PDB_structure", pdb_path)
            chains_data = process_structure(structure)
            processed_data_protein1.append(chains_data[0])
            processed_data_protein2.append(chains_data[1])

    torch.save(processed_data_protein1, protein1_path)
    torch.save(processed_data_protein2, protein2_path)

def process_structure(structure):
    model = structure[0]
    chains_data = []
    for chain in model:
        feats, coors = convert_chain_to_graph(chain, max_residues=1000)
        chains_data.append((feats, coors))
    return chains_data

def convert_chain_to_graph(chain, max_residues):
    feats, coors = [], []
    for residue in chain.get_residues()[:max_residues]:
        feats.append(extract_residue_features(residue))
        residue_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
        coors.append(np.mean(residue_coords, axis=0))

    feats = torch.tensor(np.array(feats), dtype=torch.float32)
    coors = torch.tensor(np.array(coors), dtype=torch.float32)
    return feats, coors

def extract_residue_features(residue):
    aa_types = 'ACDEFGHIKLMNPQRSTVWY'
    return [1.0 if PDB.Polypeptide.three_to_one(residue.resname.upper()) == aa else 0.0 for aa in aa_types]

# Setup data loaders
train_dataset, val_dataset, test_dataset = generate_unpaired_datasets(max_residues=1000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the network
net = EGNN_Network(
    num_tokens=21,
    num_positions=1000 * 3,
    depth=5,
    dim=8,
    num_nearest_neighbors=16,
    fourier_features=2,
    norm_coors=True,
    coor_weights_clamp_value=2.0
).to(device)

# Optimizer setup
optim = Adam(net.parameters(), lr=1e-3)

# Training and evaluation functions
def train():
    model_save_path = base_path / 'best_model.pth'
    losses_save_path = base_path / 'losses_per_epoch.txt'
    print(f"Best model will be saved to {model_save_path}")
    print(f"Losses will be saved to {losses_save_path}")

    with open(losses_save_path, 'w') as f:
        f.write('Epoch,Train Loss\n')
        for epoch in range(EPOCHS):
            net.train()
            epoch_loss = 0.0
            num_batches = 0

            for feats, coors in train_loader:
                optim.zero_grad()
                feats, coors = feats.to(device), coors.to(device)
                seqs = feats.argmax(dim=-1)
                masks = torch.ones_like(seqs).bool()
                noised_coords = coors + torch.randn_like(coors) * 0.1
                i = torch.arange(seqs.shape[1], device=device)
                adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

                _, denoised_coords = net(seqs, noised_coords, adj_mat=adj_mat, mask=masks)
                loss = F.mse_loss(denoised_coords[masks], coors[masks])
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
                num_batches += 1

            average_loss = epoch_loss / num_batches
            f.write(f'{epoch},{average_loss}\n')
            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Average Loss: {average_loss}')
                evaluate()

def evaluate():
    net.eval()
    total_val_loss = 0.0
    total_val_batches = 0

    with torch.no_grad():
        for feats, coors in val_loader:
            feats, coors = feats.to(device), coors.to(device)
            seqs = feats.argmax(dim=-1)
            masks = torch.ones_like(seqs).bool()
            i = torch.arange(seqs.shape[1], device=device)
            adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

            _, denoised_coords = net(seqs, coors, adj_mat=adj_mat, mask=masks)
            loss = F.mse_loss(denoised_coords[masks], coors[masks])
            
            total_val_loss += loss.item()
            total_val_batches += 1

        average_val_loss = total_val_loss / total_val_batches
        print(f'Validation Loss: {average_val_loss}')

train()
