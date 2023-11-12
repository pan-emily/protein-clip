import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os 
from datetime import datetime

from transformers import EsmModel, EsmTokenizer

# import local modules 
from modules import propedia, seed, clustering, model, visualization

# class PeptideReceptorDataset(Dataset):
#     def __init__(self, clusters, cluster_ids):
#         self.clusters = clusters
#         self.cluster_ids = cluster_ids

#     def __len__(self):
#         return len(self.cluster_ids)

#     def __getitem__(self, idx):
#         curr_cluster = self.clusters[self.cluster_ids[idx]]
#         curr_pair = random.choice(curr_cluster)
#         peptide_sequence = curr_pair[0]
#         receptor_sequence = curr_pair[1]
#         return peptide_sequence, receptor_sequence

import random
from torch.utils.data import Dataset

class PeptideReceptorDataset(Dataset):
    def __init__(self, clusters, cluster_ids):
        self.clusters = clusters
        self.cluster_ids = cluster_ids

    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, idx):
        # Retrieve the current cluster based on the provided index
        curr_cluster_id = self.cluster_ids[idx]
        curr_cluster = self.clusters[curr_cluster_id]

        # Check if the current cluster is empty
        if not curr_cluster:
            # Handle the case where the cluster is empty
            # You could raise an exception or handle it in a way that fits your application
            raise ValueError(f"The cluster with ID '{curr_cluster_id}' is empty.")

        # Choose a random pair from the current cluster
        curr_pair = random.choice(curr_cluster)
        peptide_sequence, receptor_sequence = curr_pair

        return peptide_sequence, receptor_sequence


def main():
    s = seed.set_seed()
    peptides, receptors = propedia.get_data()

    # Rather than get from propedia, just read from local fasta 
    # peptide_file_path = 'peptide.fasta'
    # receptor_file_path = 'receptor.fasta'
    # peptides, receptors = propedia.get_data_from_fasta(peptide_file_path, receptor_file_path)
    clusters, train_clusters, val_clusters, test_clusters = clustering.cluster(peptides, receptors, s)

    train_dataset = PeptideReceptorDataset(clusters, train_clusters)
    val_dataset = PeptideReceptorDataset(clusters, val_clusters)
    test_dataset = PeptideReceptorDataset(clusters, test_clusters)

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D").to(device)
    for param in esm_model.parameters():
        param.requires_grad = False

    # model training 
    seed.set_seed()
    input_dim = 640
    embedding_dim = 128
    h1 = 2
    h2 = 2
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model = model.ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model).to(device)
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=1e-3)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    num_epochs = 25
    training_with_grad_cache = True
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None

    print(f"Encoder loss: {-torch.tensor(1/batch_size).log().item()}")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    base_path = f'{os.getcwd()}/clip_{current_time}'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_save_path = f'{base_path}/best_model.pth'
    losses_save_path = f'{base_path}/losses_per_epoch.txt'

    print(f"Saving best models to {model_save_path}. Saving losses to {losses_save_path}")

    # visualization.plot_raw_embedding_cosine_similarities(train_loader, tokenizer, trained_model, "cpu")

    # training loop 

    with open(losses_save_path, 'w') as f:
        f.write('Epoch,Train Loss,Validation Loss\n')

        for epoch in range(25):
            if training_with_grad_cache:
                agg_batches = 1
                train_loss = model.train_gc(trained_model, train_loader, tokenizer, optimizer, device, agg_batches)
            else:
                train_loss = model.train(trained_model, train_loader, optimizer, device)
            val_loss = model.evaluate(trained_model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            f.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = trained_model.state_dict()
                torch.save(best_model_state, model_save_path)

            torch.cuda.empty_cache()
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return 


if __name__ == '__main__':
    main() 