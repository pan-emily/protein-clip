import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout_rate, esm_model):
        super(Encoder, self).__init__()
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.amino_acid_ffn = self._build_ffn(embedding_dim, h1, dropout_rate)
        self.embedding_ffn = self._build_ffn(embedding_dim, h2, dropout_rate)
        self.esm_model = esm_model

    def forward(self, seq):
        input_ids = seq['input_ids']
        attn_mask = seq['attention_mask']
        temperature = seq['temperature']
        esm_embedding = self.esm_model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        embedding = self.projection(esm_embedding)
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

class ExtendedCLIP(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout, esm_model):
        super(ExtendedCLIP, self).__init__()
        self.pep_encoder = Encoder(input_dim, embedding_dim, h1, h2, dropout, esm_model)
        self.rec_encoder = Encoder(input_dim, embedding_dim, h1, h2, dropout, esm_model)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, pep_seq, rec_seq):
        pep_seq['temperature'] = self.temperature
        rec_seq['temperature'] = self.temperature
        pep_embedding = self.pep_encoder(pep_seq)
        rec_embedding = self.rec_encoder(rec_seq)
        return pep_embedding, rec_embedding
