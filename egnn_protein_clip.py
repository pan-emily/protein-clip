import os 
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from grad_cache.functional import cached, cat_input_tensor
from transformers import EsmTokenizer, EsmModel
from egnn_pytorch import EGNN_Network
from datetime import datetime
import matplotlib.pyplot as plt

from modules import seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed.set_seed()





# model functions
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout_rate, esm_model, egnn_model):
        super(Encoder, self).__init__()
        self.esm_model = esm_model
        self.egnn_model = egnn_model
        self.projection = nn.Linear(33, embedding_dim)
        self.amino_acid_ffn = self._build_ffn(embedding_dim, h1, dropout_rate)
        self.embedding_ffn = self._build_ffn(embedding_dim, h2, dropout_rate)
        self.egnn_model.token_emb.weight = torch.nn.Parameter(
            self.esm_model.embeddings.word_embeddings.weight.t(), 
            requires_grad=False
        )

    def forward(self, seq):
        feats = seq['feats']
        mask = seq['mask']
        adj_mat = seq['adj_mat']
        coords = seq['coords']
        temperature = seq['temperature']

        egnn_embedding, _ = self.egnn_model(feats.argmax(dim=-1), coords, adj_mat=adj_mat, mask=mask.bool())
        embedding = self.projection(egnn_embedding)
        amino_acid_embedding = self.amino_acid_ffn(embedding)
        mean_embedding = self._masked_mean(amino_acid_embedding, mask)
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








# set model hyperparameters
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

egnn_model1 = EGNN_Network(
    num_tokens=input_dim,
    num_positions=1000 * 3,
    dim=33,
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
    dim=33,
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










# dataset class
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

class NewProteinDataset(Dataset):
    def __init__(self, clusters, ids):
        self.clusters = clusters
        self.cluster_ids = ids

    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, idx):
        curr_cluster = self.clusters[self.cluster_ids[idx]]
        if curr_cluster:
            sequence1 = self.clusters[idx][0]
            sequence2 = self.clusters[idx][1]
            features1, mask1 = NewProteinDataset._string_to_one_hot(''.join([chr(res_ascii) for res_ascii in sequence1[0]]))
            features2, mask2 = NewProteinDataset._string_to_one_hot(''.join([chr(res_ascii) for res_ascii in sequence2[0]]))
            coords1 = sequence1[1]
            coords2 = sequence2[1]
            return ((features1, mask1), coords1), ((features2, mask2), coords2)
        return None
    
    @staticmethod
    def _string_to_one_hot(string):
        vocab_list = (
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        "<null_1>",
        "<mask>",
        )
        vocab_index = {char: idx for idx, char in enumerate(vocab_list)}

        one_hot_tensor = torch.zeros((len(string), len(vocab_list)), dtype=torch.int32)
        mask = torch.zeros((len(string),), dtype=torch.int16)
        for i, char in enumerate(string):
            if char in vocab_index:
                one_hot_tensor[i, vocab_index[char]] = 1
                mask[i] = 1
        return one_hot_tensor, mask









# loading data
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

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)






# train functions
def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        optimizer.zero_grad()
        loss = _process_batch(model, batch_data, device, compute_grad=True)
        optimizer.step()
        total_loss += loss
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, batch_data in enumerate(data_loader):
            loss = _process_batch(model, batch_data, device)
            total_loss += loss
    return total_loss / len(data_loader)

def _process_batch(model, batch_data, device, compute_grad=False):
    sequence1, sequence2 = batch_data

    seq1 = {}
    seq1['feats'] = sequence1[0][0].to(device)
    seq1['mask'] = sequence1[0][1].to(device)
    seq1['adj_mat'] = _get_adj_mat(sequence1[0][1]).to(device)
    seq1['coords'] = sequence1[1].to(device)

    seq2 = {}
    seq2['feats'] = sequence2[0][0].to(device)
    seq2['mask'] = sequence2[0][1].to(device)
    seq2['adj_mat'] = _get_adj_mat(sequence2[0][1]).to(device)
    seq2['coords'] = sequence2[1].to(device)
    
    embedding1, embedding2 = model(seq1, seq2)
    loss = _contrastive_loss(embedding1, embedding2.t())
    if compute_grad:
        loss.backward()
    return loss.item()

def _get_adj_mat(mask):
    batch_size = mask.shape[0]
    num_nodes = mask.shape[1]
    adj_mat = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if abs(i - j) <= 1:
                adj_mat[:, i, j] = mask[:, i] & mask[:, j]
    return adj_mat

def _contrastive_loss(embedding1, embedding2):
    logits = torch.mm(embedding1, embedding2)
    exp_logits = torch.exp(logits)
    L_r = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=1)))
    L_p = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=0)))
    return (L_r + L_p) * 0.5







def train_gc(model, data_loader, optimizer, scaler, device, accumulated_batches=1):
    model.train()
    total_loss = 0

    cache_x = []
    cache_y = []
    closures_x = []
    closures_y = []

    big_batches = 0
    for step, sub_batch in enumerate(data_loader):
        sequence1, sequence2 = sub_batch

        seq1 = {}
        seq1['feats'] = sequence1[0][0].to(device)
        seq1['mask'] = sequence1[0][1].to(device)
        seq1['adj_mat'] = _get_adj_mat(sequence1[0][1]).to(device)
        seq1['coords'] = sequence1[1].to(device)
        seq1['temperature'] = model.temperature

        seq2 = {}
        seq2['feats'] = sequence2[0][0].to(device)
        seq2['mask'] = sequence2[0][1].to(device)
        seq2['adj_mat'] = _get_adj_mat(sequence2[0][1]).to(device)
        seq2['coords'] = sequence2[1].to(device)
        seq2['temperature'] = model.temperature

        rx, cx = _call_model_gc(model.encoder1, seq1)
        ry, cy = _call_model_gc(model.encoder2, seq2)

        cache_x.append(rx)
        cache_y.append(ry)
        closures_x.append(cx)
        closures_y.append(cy)

        if (step + 1) % accumulated_batches == 0:
            big_batches += 1
            print(step)
            loss = _contrastive_loss_gc(cache_x, cache_y)
            total_loss += loss.item()
            scaler.scale(loss).backward()

            for f, r in zip(closures_x, cache_x):
                f(r)
            for f, r in zip(closures_y, cache_y):
                f(r)

            cache_x = []
            cache_y = []
            closures_x = []
            closures_y = []

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / big_batches

@cached
@autocast()
def _call_model_gc(model, input):
    return model(input)

@cat_input_tensor
@autocast()
def _contrastive_loss_gc(x, y):
    logits = torch.matmul(x, y.transpose(0, 1))
    exp_logits = torch.exp(logits)
    L_r = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=1)))
    L_p = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=0)))
    loss = (L_r + L_p) * 0.5
    return loss







def plot_embedding_cosine_similarities(base_path, title, data_loader, model, device):
    sequence1, sequence2 =  next(iter(data_loader))

    seq1 = {}
    seq1['feats'] = sequence1[0][0].to(device)
    seq1['mask'] = sequence1[0][1].to(device)
    seq1['adj_mat'] = _get_adj_mat(sequence1[0][1]).to(device)
    seq1['coords'] = sequence1[1].to(device)

    seq2 = {}
    seq2['feats'] = sequence2[0][0].to(device)
    seq2['mask'] = sequence2[0][1].to(device)
    seq2['adj_mat'] = _get_adj_mat(sequence2[0][1]).to(device)
    seq2['coords'] = sequence2[1].to(device)
    
    embedding1, embedding2 = model(seq1, seq2)
    similarity_matrix = _compute_embedding_cosine_similarities(model, embedding1, embedding2)
    similarity_matrix_np = similarity_matrix.cpu().detach().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(similarity_matrix_np, cmap="ocean", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Sequence 1")
    plt.ylabel("Sequence 2")
    plot_path = _save_plot(base_path)
    print(f"{title} plot saved to {plot_path}")

def _compute_embedding_cosine_similarities(model, embedding1, embedding2):
    similarity_matrix = torch.mm(embedding1, embedding2.t())
    return similarity_matrix * torch.exp(-model.temperature)

def plot_loss_curves(base_path, train_losses, val_losses, train_batch_size, val_batch_size):
    title = 'Training and Validation Loss Relative to Random'
    plt.plot([i/-torch.tensor(1/train_batch_size).log().item() for i in train_losses], label='Train Loss')
    plt.plot([i/-torch.tensor(1/val_batch_size).log().item() for i in val_losses], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Relative to Random')
    plt.legend()
    plot_path = _save_plot(base_path)
    print(f"{title} plot saved to {plot_path}")

def _save_plot(base_path, fig_num=[1]):
    folder_path = os.path.join(base_path, "figures")
    os.makedirs(folder_path, exist_ok=True)
    filename = f"figure {fig_num[0]}.png"
    plot_path = os.path.join(folder_path, filename)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()
    fig_num[0] += 1
    return plot_path







# training script
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
base_path = f'{os.getcwd()}/runs/{timestamp}'
os.makedirs(base_path, exist_ok=True)
print(f"All run info will be saved to {base_path}")
num_epochs = 25
optimizer = Adam(trained_model.parameters(), lr=1e-3)
training_with_grad_cache = True
if training_with_grad_cache:
    scaler = GradScaler()
    accumulated_batches = 32

train_losses, val_losses = [], []
best_val_loss = float('inf')
best_model_state = None
plot_embedding_cosine_similarities(base_path, "Raw Embedding Cosine Similarities", train_loader, trained_model, device)
model_save_path = f'{base_path}/best_model.pth'
losses_save_path = f'{base_path}/losses_per_epoch.txt'
print(f"Best model will be saved to {model_save_path}")
print(f"Losses will be saved to {losses_save_path}")

with open(losses_save_path, 'w') as f:
    f.write('Epoch,Train Loss,Validation Loss\n')

    for epoch in range(num_epochs):
        if training_with_grad_cache:
            train_loss = train_gc(trained_model, train_loader, optimizer, scaler, device, accumulated_batches)
        else:
            train_loss = train(trained_model, train_loader, optimizer, device)
        val_loss = evaluate(trained_model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        f.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = trained_model.state_dict()
            torch.save(best_model_state, model_save_path)
            best_trained_model = ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model, egnn_model1, egnn_model2).to(device)
            best_trained_model.load_state_dict(torch.load(model_save_path))

        plot_embedding_cosine_similarities(base_path, f"Trained Embedding Cosine Similarities on Train Set - Epoch {epoch + 1}", train_loader, best_trained_model, device)    
        plot_embedding_cosine_similarities(base_path, f"Trained Embedding Cosine Similarities on Val Set - Epoch {epoch + 1}", val_loader, best_trained_model, device)

        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if training_with_grad_cache:
    plot_loss_curves(base_path, train_losses, val_losses, batch_size*accumulated_batches, batch_size)
else:
    plot_loss_curves(base_path, train_losses, val_losses, batch_size, batch_size)
