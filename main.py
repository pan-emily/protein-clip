import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from transformers import EsmModel, EsmTokenizer

# import local modules 
from modules import propedia, seed, clustering, model

class PeptideReceptorDataset(Dataset):
    def __init__(self, clusters, cluster_ids):
        self.clusters = clusters
        self.cluster_ids = cluster_ids

    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, idx):
        curr_cluster = self.clusters[self.cluster_ids[idx]]
        curr_pair = random.choice(curr_cluster)
        peptide_sequence = curr_pair[0]
        receptor_sequence = curr_pair[1]
        return peptide_sequence, receptor_sequence

def main():
    s = seed.set_seed()
    peptides, receptors = propedia.get_data()
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

    print(f"loss: {-torch.tensor(1/batch_size).log().item()}")


    return 


if __name__ == '__main__':
    main() 