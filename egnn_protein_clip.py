from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from transformers import EsmTokenizer, EsmModel
from egnn_pytorch import EGNN_Network
from modules import seed, visualizations
import os 
from datetime import datetime



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed.set_seed()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
base_path = f'{os.getcwd()}/runs/{timestamp}'
os.makedirs(base_path, exist_ok=True)
print(f"All run info will be saved to {base_path}")


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout_rate, esm_model, egnn_model):
        super(Encoder, self).__init__()
        self.esm_model = esm_model
        self.egnn_model = egnn_model
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.amino_acid_ffn = self._build_ffn(embedding_dim, h1, dropout_rate)
        self.embedding_ffn = self._build_ffn(embedding_dim, h2, dropout_rate)

    def forward(self, seq):
        input_ids = seq['input_ids']
        attn_mask = seq['attention_mask']
        temperature = seq['temperature']
        coords = seq['coords']
        batch_size = coords.shape[0]
        num_nodes = coords.shape[1]

        # get feat embeddings from esm
        esm_embedding = self.esm_model.embeddings(input_ids=input_ids, attention_mask=attn_mask)

        # send to egnn with coords 
        egnn_embedding, _ = egnn_model(esm_embedding, coords, adj_mat=_get_adj_mat(batch_size, num_nodes), mask=attn_mask)

        embedding = self.projection(egnn_embedding)
        amino_acid_embedding = self.amino_acid_ffn(embedding)
        mean_embedding = self._masked_mean(amino_acid_embedding, attn_mask)
        embedding_output = self.embedding_ffn(mean_embedding)
        normed_embedding = F.normalize(embedding_output, dim=-1)
        scaled_embedding = normed_embedding * torch.exp(temperature / 2)
        return scaled_embedding
    
    def _build_ffn(self, embedding_dim, depth, dropout_rate):
        layers = [nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout_rate)
        ) for _ in range(depth - 1)]
        layers.append(nn.Linear(embedding_dim, embedding_dim))
        return nn.Sequential(*layers)
    
    @staticmethod
    def _masked_mean(h, mask):
        sum_masked_h = torch.sum(h * mask.unsqueeze(-1), dim=1)
        count_non_masked = mask.sum(dim=1, keepdim=True).clamp_(min=1)
        mean_masked_h = sum_masked_h.div_(count_non_masked)
        return mean_masked_h

    @staticmethod
    def _get_adj_mat(batch_size, num_nodes):
        adj_mat = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if abs(i - j) <= 1:
                    adj_mat[:, i, j] = mask[:, i] & mask[:, j]
        return adj_mat

class ExtendedCLIP(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout, esm_model, egnn_model1, egnn_model2):
        super(ExtendedCLIP, self).__init__()
        self.encoder1 = Encoder(input_dim, embedding_dim, h1, h2, dropout, esm_model, egnn_model1)
        self.encoder2 = Encoder(input_dim, embedding_dim, h1, h2, dropout, esm_model, egnn_model2)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, seq1, seq2):
        seq1['temperature'] = self.temperature
        seq2['temperature'] = self.temperature
        embedding1 = self.encoder1(seq1)
        embedding2 = self.encoder2(seq2)
        return embedding1, embedding2


tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
esm_model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
for param in esm_model.parameters():
    param.requires_grad = False
input_dim = 640


# checking esm embedding
# sequence = ["XXX", "AAAAA"]
# encoded_seq = tokenizer(sequence, return_tensors="pt", padding=True)
# hidden_states = esm_model.embeddings(**encoded_seq)
# print(hidden_states)
# print(encoded_seq["attention_mask"])
# print(hidden_states.shape) #shape: (1, start + seq + end + 0s with bool mask in encoded_seq['attention_mask'], input_dim)


# set model hyperparameters
egnn_model1 = EGNN_Network(
    num_tokens=input_dim,
    num_positions=1000 * 3,
    dim=32,
    depth=5,
    num_nearest_neighbors=16, #maybe ignored 
    fourier_features=2,
    norm_coors=True,
    update_feats=True,
    update_coors=False,
    coor_weights_clamp_value=2.0
)
egnn_model2 = EGNN_Network(
    num_tokens=input_dim,
    num_positions=1000 * 3,
    dim=32,
    depth=5,
    num_nearest_neighbors=16, #maybe ignored 
    fourier_features=2,
    norm_coors=True,
    update_feats=True,
    update_coors=False,
    coor_weights_clamp_value=2.0
)

esm_model = esm_model.to(device)
egnn_model1 = egnn_model1.to(device)
egnn_model2 = egnn_model2.to(device)

embedding_dim = 128
h1 = 2
h2 = 2
dropout = 0.1
trained_model = ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model, egnn_model1, egnn_model2)
trained_model = trained_model.to(device)

data_dir = Path('data')
protein1_file_path = data_dir / 'protein1.pt'
protein2_file_path = data_dir / 'protein2.pt'
protein1s = torch.load(protein1_file_path)
protein2s = torch.load(protein2_file_path)











from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random


class NewProteinDataset(Dataset):
    def __init__(self, clusters, ids):
        self.clusters = clusters
        self.cluster_ids = ids
        self.aa_types = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, idx):
        curr_cluster = self.clusters[self.cluster_ids[idx]]
        if curr_cluster:
            sequence1 = self.clusters[idx][0]
            sequence2 = self.clusters[idx][1]
            features1 = self._to_tensor(sequence1[0], dtype=torch.long, device=device)
            coords1 = self._to_tensor(sequence1[1], dtype=torch.float, device=device)
            features2 = self._to_tensor(sequence2[0], dtype=torch.long, device=device)
            coords2 = self._to_tensor(sequence2[1], dtype=torch.float, device=device)
            # Convert one-hot encoded features to letters
            features1 = [self.aa_types[torch.argmax(feature)] if torch.any(feature) else 'X' for feature in features1]
            features2 = [self.aa_types[torch.argmax(feature)] if torch.any(feature) else 'X' for feature in features2]
            # check if any other values other than 0 in features1
            # turn features into string and make string tensor to be on device
            features1 = torch.tensor(''.join(features1), dtype=torch.str, device=device)
            features2 = torch.tensor(''.join(features2), dtype=torch.str, device=device)
        return None

    @staticmethod
    def _to_tensor(data, dtype, device):
        return data.clone().detach().to(dtype=dtype, device=device) if torch.is_tensor(data) else torch.tensor(data, dtype=dtype, device=device)


clusters = {}
for i, e in enumerate(zip(protein1s, protein2s)):
    clusters[i] = e

cluster_ids = list(clusters.keys())
random.shuffle(cluster_ids) 

num_train = int(0.7 * len(cluster_ids))
num_val = int(0.15 * len(cluster_ids))

train_clusters = cluster_ids[:num_train]
val_clusters = cluster_ids[num_train:num_train+num_val]
test_clusters = cluster_ids[num_train+num_val:]

train_dataset = NewProteinDataset(clusters, train_clusters)
val_dataset = NewProteinDataset(clusters, val_clusters)
test_dataset = NewProteinDataset(clusters, test_clusters)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)









def train(model, data_loader, optimizer, tokenizer, device):
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        optimizer.zero_grad()
        loss = _process_batch(model, batch_data, tokenizer, device, compute_grad=True)
        optimizer.step()
        total_loss += loss
    return total_loss / len(data_loader)

def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, batch_data in enumerate(data_loader):
            loss = _process_batch(model, batch_data, tokenizer, device)
            total_loss += loss
    return total_loss / len(data_loader)

def _process_batch(model, batch_data, tokenizer, device, compute_grad=False):
    sequence1, sequence2 = batch_data
    breakpoint()
    sequence1 = tokenizer(sequence1[0], return_tensors='pt', padding=True).to(device)
    sequence2 = tokenizer(sequence2[0], return_tensors='pt', padding=True).to(device)
    sequence1['coords'] = sequence1[1]
    sequence2['coords'] = sequence2[1]
    embedding1, embedding2 = model(sequence2, sequence2)
    loss = _contrastive_loss(embedding1, embedding2.t())
    if compute_grad:
        loss.backward()
    return loss.item()

def _contrastive_loss(embedding1, embedding2):
    logits = torch.mm(embedding1, embedding2)
    exp_logits = torch.exp(logits)
    L_r = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=1)))
    L_p = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=0)))
    return (L_r + L_p) * 0.5


num_epochs = 20
optimizer = Adam(trained_model.parameters(), lr=1e-3)

# init before training
train_losses, val_losses = [], []
best_val_loss = float('inf')
best_model_state = None
"""
(protein-clip) �  protein-clip git:(continuation-spring-2024) � python egnn_protein_clip.py
All run info will be saved to /home/ubuntu/protein-clip/runs/20240423_062624_576605
Some weights of EsmModel were not initialized from the model checkpoint at facebook/esm2_t30_150M_UR50D and are newly initialized: ['esm.pooler.dense.bias', 'esm.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Traceback (most recent call last):
  File "/home/ubuntu/protein-clip/egnn_protein_clip.py", line 269, in <module>
    visualizations.plot_embedding_cosine_similarities(base_path, "Raw Embedding Cosine Similarities", train_loader, tokenizer, trained_model, device)
  File "/home/ubuntu/protein-clip/modules/visualizations.py", line 23, in plot_embedding_cosine_similarities
    curr_peptides = tokenizer(curr_peptides, return_tensors='pt', padding=True).to(device)
  File "/home/ubuntu/miniconda3/envs/protein-clip/lib/python3.10/site-packages/transformers/tokenization_utils_base.py", line 2602, in __call__
    encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
  File "/home/ubuntu/miniconda3/envs/protein-clip/lib/python3.10/site-packages/transformers/tokenization_utils_base.py", line 2660, in _call_one
    raise ValueError(
ValueError: text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
"""
# visualizations.plot_embedding_cosine_similarities(base_path, "Raw Embedding Cosine Similarities", train_loader, tokenizer, trained_model, device)
model_save_path = f'{base_path}/best_model.pth'
losses_save_path = f'{base_path}/losses_per_epoch.txt'
print(f"Best model will be saved to {model_save_path}")
print(f"Losses will be saved to {losses_save_path}")

# training 
with open(losses_save_path, 'w') as f:
    f.write('Epoch,Train Loss,Validation Loss\n')

    for epoch in range(num_epochs):
        print('new epoch')
        train_loss = train(trained_model, train_loader, optimizer, tokenizer, device)
        val_loss = evaluate(trained_model, val_loader, tokenizer, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        f.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = trained_model.state_dict()
            torch.save(best_model_state, model_save_path)
            best_trained_model = ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model, egnn_model1, egnn_model2).to(device)
            best_trained_model.load_state_dict(torch.load(model_save_path))

        # visualizations.plot_embedding_cosine_similarities(base_path, f"Trained Embedding Cosine Similarities on Train Set - Epoch {epoch + 1}", train_loader, tokenizer, best_trained_model, device)    
        # visualizations.plot_embedding_cosine_similarities(base_path, f"Trained Embedding Cosine Similarities on Val Set - Epoch {epoch + 1}", val_loader, tokenizer, best_trained_model, device)

