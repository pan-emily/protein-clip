import torch 
import torch.nn as nn
import torch.nn.functional as F

from grad_cache.functional import cached, cat_input_tensor
from torch.cuda.amp import GradScaler, autocast

scaler = torch.cuda.amp.GradScaler()

def masked_mean(h, mask):
    sum_masked_h = torch.sum(h * mask.unsqueeze(-1), dim=1)
    count_non_masked = mask.sum(dim=1, keepdim=True).clamp_(min=1)
    mean_masked_h = sum_masked_h.div_(count_non_masked)
    return mean_masked_h

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout_rate, esm_model):
        super(Encoder, self).__init__()
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.amino_acid_ffn = self._build_ffn(embedding_dim, h1, dropout_rate)
        self.embedding_ffn = self._build_ffn(embedding_dim, h2, dropout_rate)
        self.esm_model = esm_model

    def _build_ffn(self, embedding_dim, depth, dropout_rate):
        layers = [nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout_rate)
        ) for _ in range(depth - 1)]
        layers.append(nn.Linear(embedding_dim, embedding_dim))
        return nn.Sequential(*layers)

    def forward(self, seq):
        input_ids = seq['input_ids']
        attn_mask = seq['attention_mask']
        temperature = seq['temperature']
        esm_embedding = self.esm_model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        embedding = self.projection(esm_embedding)
        amino_acid_embedding = self.amino_acid_ffn(embedding)
        mean_embedding = masked_mean(amino_acid_embedding, attn_mask)
        embedding_output = self.embedding_ffn(mean_embedding)
        normed_embedding = F.normalize(embedding_output, dim=-1)
        scaled_embedding = normed_embedding * torch.exp(temperature / 2)
        return scaled_embedding

class ExtendedCLIP(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout, esm_model):
        super(ExtendedCLIP, self).__init__()
        self.pep_encoder = Encoder(input_dim, embedding_dim, h1, h2, dropout, esm_model)
        self.rec_encoder = Encoder(input_dim, embedding_dim, h1, h2, dropout, esm_model)
        self.t = nn.Parameter(torch.tensor(1.0))

    def forward(self, pep_seq, rec_seq):
        pep_seq['temperature'] = self.t
        rec_seq['temperature'] = self.t
        pep_embedding = self.pep_encoder(pep_seq)
        rec_embedding = self.rec_encoder(rec_seq)
        return pep_embedding, rec_embedding

def compute_similarity_matrix(model, peptides, receptors):
    pep_norm_embedding, rec_norm_embedding = model(peptides, receptors)
    similarity_matrix = torch.mm(pep_norm_embedding, rec_norm_embedding.t())
    return similarity_matrix * torch.exp(-model.t)

def compute_loss(pep_embedding, rec_embedding):
    logits = torch.mm(pep_embedding, rec_embedding)
    exp_logits = torch.exp(logits)
    L_r = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=1)))
    L_p = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=0)))
    return (L_r + L_p) * 0.5

def process_batch(model, batch_data, device, tokenizer, compute_grad=False):
    print('batch')
    peptides, receptors = batch_data
    peptides = tokenizer(peptides, return_tensors='pt', padding=True).to(device)
    receptors = tokenizer(receptors, return_tensors='pt', padding=True).to(device)
    pep_embedding, rec_embedding = model(peptides, receptors)
    loss = compute_loss(pep_embedding, rec_embedding.t())
    if compute_grad:
        loss.backward()
    return loss.item()

def train(model, data_loader, optimizer, tokenizer, device):
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        optimizer.zero_grad()
        loss = process_batch(model, batch_data, device, tokenizer, compute_grad=True)
        optimizer.step()
        total_loss += loss
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device, tokenizer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data in data_loader:
            loss = process_batch(model, batch_data, device, tokenizer)
            total_loss += loss
    return total_loss / len(data_loader)

@cached
@autocast()
def call_model(model, input):
    return model(input)

@cat_input_tensor
@autocast()
def contrastive_loss(x, y):
    logits = torch.matmul(x, y.transpose(0, 1))
    exp_logits = torch.exp(logits)
    L_r = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=1)))
    L_p = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=0)))
    loss = (L_r + L_p) * 0.5
    return loss

def train_gc(model, data_loader, tokenizer, optimizer, device, agg_batches=2):
    model.train()
    total_loss = 0

    cache_x = []
    cache_y = []
    closures_x = []
    closures_y = []

    big_batches = 0
    for step, sub_batch in enumerate(data_loader):
        print(f"batch {step}")
        xx, yy = sub_batch
        xx = tokenizer(xx, return_tensors='pt', padding=True).to(device)
        yy = tokenizer(yy, return_tensors='pt', padding=True).to(device)
        xx['temperature'] = model.t
        yy['temperature'] = model.t

        rx, cx = call_model(model.pep_encoder, xx)
        ry, cy = call_model(model.rec_encoder, yy)

        cache_x.append(rx)
        cache_y.append(ry)
        closures_x.append(cx)
        closures_y.append(cy)

        if (step + 1) % agg_batches == 0:
            big_batches += 1
            loss = contrastive_loss(cache_x, cache_y)
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



