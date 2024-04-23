import torch
from torch import nn
from transformers import EsmTokenizer, EsmModel
from egnn_pytorch import EGNN_Network

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
sequence = ["XXX", "AAAAA"]
encoded_seq = tokenizer(sequence, return_tensors="pt", padding=True)
hidden_states = esm_model.embeddings(**encoded_seq)
print(hidden_states)
print(encoded_seq["attention_mask"])
print(hidden_states.shape) #shape: (1, start + seq + end + 0s with bool mask in encoded_seq['attention_mask'], input_dim)


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

embedding_dim = 128
h1 = 2
h2 = 2
dropout = 0.1
trained_model = ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model, egnn_model1, egnn_model2)


