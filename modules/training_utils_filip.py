import torch 
from einops import reduce

def _contrastive_loss(logits: torch.Tensor, use_dcl: bool = False):
    exp_logits = logits.exp()
    loss_numerator = exp_logits.diagonal()

    if use_dcl:
        pos_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        exp_logits = exp_logits.masked_fill(pos_mask, 0)

    loss_denominator = reduce(exp_logits, "b t -> b", "sum")
    return (-loss_numerator.log() + loss_denominator.log()).mean()


def _compute_loss(sim_scores_A, sim_scores_B):
    loss_A = _contrastive_loss(sim_scores_A, use_dcl=True)
    loss_B = _contrastive_loss(sim_scores_B, use_dcl=True)
    return (loss_A + loss_B) / 2

def _process_batch(model, batch_data, tokenizer, device, compute_grad=False):
    peptides, receptors = batch_data
    peptides = tokenizer(peptides, return_tensors='pt', padding=True).to(device)
    receptors = tokenizer(receptors, return_tensors='pt', padding=True).to(device)

    sim_scores_A, sim_scores_B = model(peptides, receptors)
    loss = _compute_loss(sim_scores_A, sim_scores_B)

    if compute_grad:
        loss.backward()
    return loss.item()

def train(model, data_loader, optimizer, tokenier, device):
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        optimizer.zero_grad()
        loss = _process_batch(model, batch_data, tokenier, device, compute_grad=True)
        optimizer.step()
        total_loss += loss
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data in data_loader:
            loss = _process_batch(model, batch_data, device)
            total_loss += loss
    return total_loss / len(data_loader)
