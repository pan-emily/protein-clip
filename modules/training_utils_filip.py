import torch 
from einops import reduce

def _contrastive_loss(logits: torch.Tensor, use_dcl: bool = False):
    """
    Compute the contrastive loss.

    Args:
        logits (torch.Tensor): Pairwise similarity logits.
        use_dcl (bool): Whether to use the Diagonal-Corrected Loss (DCL). Default is False.

    Returns:
        torch.Tensor: Contrastive loss.
    """
    exp_logits = logits.exp()
    loss_numerator = exp_logits.diagonal()

    if use_dcl:
        pos_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        exp_logits = exp_logits.masked_fill(pos_mask, 0)

    loss_denominator = reduce(exp_logits, "b t -> b", "sum")
    return (-loss_numerator.log() + loss_denominator.log()).mean()


def _compute_loss(sim_scores_A, sim_scores_B):
    """
    Compute the loss based on similarity scores from modalities A and B.

    Args:
        sim_scores_A (torch.Tensor): Similarity scores for modality A.
        sim_scores_B (torch.Tensor): Similarity scores for modality B.

    Returns:
        torch.Tensor: Combined loss.
    """
    loss_A = _contrastive_loss(sim_scores_A, use_dcl=False)
    loss_B = _contrastive_loss(sim_scores_B, use_dcl=False)
    return (loss_A + loss_B) / 2

def _process_batch(model, batch_data, tokenizer, device, compute_grad=False):
    """
    Process a batch of data using the model and compute the loss.

    Args:
        model: The model to be used for processing.
        batch_data (tuple): A tuple of peptides and receptors for the batch.
        tokenizer: Tokenizer for encoding peptides and receptors.
        device: Device to place the data and model on.
        compute_grad (bool): Whether to compute gradients. Default is False.

    Returns:
        float: Loss value for the batch.
    """
    peptides, receptors = batch_data
    peptides = tokenizer(peptides, return_tensors='pt', padding=True).to(device)
    receptors = tokenizer(receptors, return_tensors='pt', padding=True).to(device)

    sim_scores_A, sim_scores_B = model(peptides, receptors)
    loss = _compute_loss(sim_scores_A, sim_scores_B)

    if compute_grad:
        loss.backward()
    return loss.item()

def train(model, data_loader, optimizer, tokenier, device):
    """
    Train the model on the given data.

    Args:
        model: The model to be trained.
        data_loader: DataLoader for the training data.
        optimizer: Optimizer for updating model parameters.
        tokenizer: Tokenizer for encoding peptides and receptors.
        device: Device to place the data and model on.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        optimizer.zero_grad()
        loss = _process_batch(model, batch_data, tokenier, device, compute_grad=True)
        optimizer.step()
        total_loss += loss
    return total_loss / len(data_loader)

def evaluate(model, data_loader, tokenizer, device):
    """
    Evaluate the model on the given data.

    Args:
        model: The model to be evaluated.
        data_loader: DataLoader for the evaluation data.
        tokenizer: Tokenizer for encoding peptides and receptors.
        device: Device to place the data and model on.

    Returns:
        float: Average evaluation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data in data_loader:
            loss = _process_batch(model, batch_data, tokenizer, device)
            total_loss += loss
    return total_loss / len(data_loader)
