import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from egnn_pytorch import EGNN_Network
from modules import data_utils_fubody
from pathlib import Path
from datetime import datetime
import os

torch.set_default_dtype(torch.float64)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
base_path = f'{os.getcwd()}/runs/{timestamp}'
os.makedirs(base_path, exist_ok=True)
print(f"All run info will be saved to {base_path}")

BATCH_SIZE = 8
EPOCHS = 1000

train_dataset, val_dataset, test_dataset = data_utils_fubody.generate_unpaired_datasets(max_residues=1000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

data_dir = Path('data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = EGNN_Network(
    num_tokens=21,
    num_positions=1000*3,
    depth=5,
    dim=8,
    num_nearest_neighbors=16,
    fourier_features=2,
    norm_coors=True,
    coor_weights_clamp_value=2.
).to(device)

optim = Adam(net.parameters(), lr=1e-3)

def train():
    model_save_path = f'{base_path}/best_model.pth'
    losses_save_path = f'{base_path}/losses_per_epoch.txt'
    print(f"Best model will be saved to {model_save_path}")
    print(f"Losses will be saved to {losses_save_path}")
    
    with open(losses_save_path, 'w') as f:

        f.write('Epoch,Train Loss\n')
        for epoch in range(EPOCHS):
            net.train()
            epoch_loss = 0.0
            num_batches = 0

            for feats, coors in train_loader:
                feats, coors = feats.to(device), coors.to(device)

                seqs = feats.argmax(dim=-1)
                masks = torch.ones_like(seqs).bool()
                noised_coords = coors + torch.randn_like(coors) * 0.1
                i = torch.arange(seqs.shape[1], device=seqs.device)
                adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

                _, denoised_coords = net(seqs, noised_coords, adj_mat=adj_mat, mask=masks)

                loss = F.mse_loss(denoised_coords[masks], coors[masks])
                loss.backward()
                optim.step()
                optim.zero_grad()

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
            i = torch.arange(seqs.shape[1], device=seqs.device)
            adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

            _, denoised_coords = net(seqs, coors, adj_mat=adj_mat, mask=masks)
            loss = F.mse_loss(denoised_coords[masks], coors[masks])
            
            total_val_loss += loss.item()
            total_val_batches += 1
        
        average_val_loss = total_val_loss / total_val_batches
        print(f'Validation Loss: {average_val_loss}')

train()
