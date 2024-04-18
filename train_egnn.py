import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from egnn_pytorch import EGNN_Network
from modules import data_utils_fubody
from pathlib import Path

torch.set_default_dtype(torch.float64)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 1000  # Set the number of epochs according to your need

# Load your processed data
train_dataset, val_dataset, test_dataset = data_utils_fubody.generate_unpaired_datasets(max_residues=1000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

data_dir = Path('data')

# Check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = EGNN_Network(
    num_tokens=21,
    num_positions=1000*3,  # Adjust based on your data, 1000 is the max_residues in your pre-processing script
    depth=5,
    dim=8,
    num_nearest_neighbors=16,
    fourier_features=2,
    norm_coors=True,
    coor_weights_clamp_value=2.
).to(device)

optim = Adam(net.parameters(), lr=1e-3)

def train(): #wagmi
    for epoch in range(EPOCHS):
        net.train()
        for feats, coors in train_loader:
            feats, coors = feats.to(device), coors.to(device)

            # Process features and coordinates
            seqs = feats.argmax(dim=-1)
            masks = torch.ones_like(seqs).bool()

            # Adding noise
            noised_coords = coors + torch.randn_like(coors) * 0.1

            # Adjacency matrix calculation
            i = torch.arange(seqs.shape[1], device=seqs.device)
            adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

            _, denoised_coords = net(seqs, noised_coords, adj_mat=adj_mat, mask=masks)

            loss = F.mse_loss(denoised_coords[masks], coors[masks])
            loss.backward()
            optim.step()
            optim.zero_grad()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                evaluate()

def evaluate():
    net.eval()
    with torch.no_grad():
        for feats, coors in val_loader:
            feats, coors = feats.to(device), coors.to(device)

            seqs = feats.argmax(dim=-1)
            masks = torch.ones_like(seqs).bool()

            i = torch.arange(seqs.shape[1], device=seqs.device)
            adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

            _, denoised_coords = net(seqs, coors, adj_mat=adj_mat, mask=masks)

            loss = F.mse_loss(denoised_coords[masks], coors[masks])
            print(f'Validation Loss: {loss.item()}')

train()