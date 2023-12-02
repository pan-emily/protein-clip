import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, einsum, rearrange


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

class FILIPEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout_rate, esm_model):
        super(FILIPEncoder, self).__init__()
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
        return amino_acid_embedding, attn_mask

class ExtendedFILIP(nn.Module):
    def __init__(self, input_dim, embedding_dim, h1, h2, dropout, esm_model):
        super(ExtendedFILIP, self).__init__()
        self.pep_encoder = FILIPEncoder(input_dim, embedding_dim, h1, h2, dropout, esm_model)
        self.rec_encoder = FILIPEncoder(input_dim, embedding_dim, h1, h2, dropout, esm_model)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, pep_seq, rec_seq):
        pep_seq['temperature'] = self.temperature
        rec_seq['temperature'] = self.temperature
        pep_embedding, pep_mask = self.pep_encoder(pep_seq)
        rec_embedding, rec_mask = self.rec_encoder(rec_seq)

        # normalize embeddings
        pep_embedding = pep_embedding / pep_embedding.norm(dim=-1, keepdim=True)
        rec_embedding = rec_embedding / rec_embedding.norm(dim=-1, keepdim=True)

        sim_scores_A, sim_scores_B = self._filip_similarity_score(
            pep_embedding, rec_embedding, pep_mask, rec_mask, self.temperature
        )

        return sim_scores_A, sim_scores_B

    @staticmethod
    def _masked_mean(t, mask, dim=1, eps=1e-6):
        t = t.masked_fill(~mask, 0.0)
        numer = t.sum(dim=dim)
        denom = mask.sum(dim=dim).clamp(min=eps)
        return numer / denom

    @staticmethod
    def _mean_average_similarity_score(
        hA: torch.Tensor, # B, T1, D
        hB: torch.Tensor, # B, T2, D
        maskA: torch.Tensor, # B, T1
        maskB: torch.Tensor, # B, T2
        temperature: torch.float
    ):
        hA = reduce(hA * maskA[..., None], "b t d -> b d", "mean")
        hB = reduce(hB * maskB[..., None], "b t d -> b d", "mean")
        logits = einsum(hA, hB, "b1 d, b2 d -> b1 b2") / temperature
        return logits

    @staticmethod
    def filip_similarity_score(
        hA: torch.Tensor,
        hB: torch.Tensor,
        maskA: torch.Tensor,
        maskB: torch.Tensor,
        temperature: torch.float,
        include_group: bool = False
    ):
        """Computes a filip similarity score (described in
        https://arxiv.org/pdf/2111.07783.pdf) for modalities A and B.
        # recombinase clusters, though I suspect this may only be true on the protein side.
        hA:     group, batch, time, dim
        hB:     group, batch, time, dim
        maskA:  group, batch, time
        maskB:  group, batch, time
        """
        # Don't do separate cases for each of these: einops will throw an error if you include
        # a group when you said you don't want to, and this is by design
        maskA = maskA.bool()
        maskB = maskB.bool()

        if not include_group:
            hA = rearrange(hA, "b t d -> 1 b t d")
            hB = rearrange(hB, "b t d -> 1 b t d")
            maskA = rearrange(maskA, "b t -> 1 b t")
            maskB = rearrange(maskB, "b t -> 1 b t")

        sim_scores = einsum(hA, hB, "m bA tA d, n bB tB d -> m n bA bB tA tB") / temperature
        maskA = rearrange(maskA, "m bA tA -> m 1 bA 1 tA 1")
        maskB = rearrange(maskB, "n bB tB -> 1 n 1 bB 1 tB")
        combined_mask = maskA * maskB

        # Mask out all padding tokens for both modalities
        sim_scores_masked = sim_scores.masked_fill(
            ~combined_mask, torch.finfo(sim_scores.dtype).min
        )

        sim_scores_A = reduce(sim_scores_masked, "... tA tB -> ... tA", "max")
        sim_scores_B = reduce(sim_scores_masked, "... tA tB -> ... tB", "max")

        sim_scores_A = ExtendedFILIP._masked_mean(
            sim_scores_A, rearrange(maskA, "... tA 1 -> ... tA"), dim=-1
        )

        sim_scores_B = ExtendedFILIP._masked_mean(
            sim_scores_B, rearrange(maskB, "... 1 tB -> ... tB"), dim=-1
        )

        if not include_group:
            sim_scores_A, sim_scores_B = map(lambda s: rearrange(s, "1 1 b t -> b t"), (sim_scores_A, sim_scores_B))

        return sim_scores_A, sim_scores_B