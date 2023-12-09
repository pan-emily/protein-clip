import torch 
from grad_cache.functional import cached, cat_input_tensor
from torch.cuda.amp import autocast
import numpy as np


def train(model, data_loader, optimizer, tokenizer, device):
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
        loss = _process_batch(model, batch_data, tokenizer, device, compute_grad=True)
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
        for step, batch_data in enumerate(data_loader):
            loss = _process_batch(model, batch_data, tokenizer, device)
            total_loss += loss
    return total_loss / len(data_loader)

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
    pep_embedding, rec_embedding = model(peptides, receptors)
    loss = _contrastive_loss(pep_embedding, rec_embedding.t())
    if compute_grad:
        loss.backward()
    return loss.item()

def _contrastive_loss(pep_embedding, rec_embedding):
    """
    Compute the contrastive loss for peptide and receptor embeddings.

    Args:
        pep_embedding (torch.Tensor): Peptide embeddings.
        rec_embedding (torch.Tensor): Receptor embeddings.

    Returns:
        torch.Tensor: Contrastive loss.
    """
    logits = torch.mm(pep_embedding, rec_embedding)
    exp_logits = torch.exp(logits)
    L_r = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=1)))
    L_p = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=0)))
    return (L_r + L_p) * 0.5


def train_gc(model, data_loader, tokenizer, optimizer, scaler, device, accumulated_batches=1):
    """
    Train the model with gradient caching and accumulation.

    Args:
        model: The model to be trained.
        data_loader: DataLoader for the training data.
        tokenizer: Tokenizer for encoding peptides and receptors.
        optimizer: Optimizer for updating model parameters.
        scaler: Gradient scaler for mixed-precision training.
        device: Device to place the data and model on.
        accumulated_batches (int): Number of batches to accumulate gradients over.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0

    cache_x = []
    cache_y = []
    closures_x = []
    closures_y = []

    big_batches = 0
    for step, sub_batch in enumerate(data_loader):
        xx, yy = sub_batch

        xx = tokenizer(xx, return_tensors='pt', padding=True).to(device)
        yy = tokenizer(yy, return_tensors='pt', padding=True).to(device)

        xx['temperature'] = model.temperature
        yy['temperature'] = model.temperature

        rx, cx = _call_model_gc(model.pep_encoder, xx)
        ry, cy = _call_model_gc(model.rec_encoder, yy)

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

@cat_input_tensor
@autocast()
def get_logits(x, y):
    """
    Compute logits for given input tensors.

    Args:
        x: Input tensor x.
        y: Input tensor y.

    Returns:
        torch.Tensor: Logits tensor.
    """
    logits = torch.matmul(x, y.transpose(0, 1))
    exp_logits = torch.exp(logits)
    return exp_logits

def eval_gc_allrec_onepep(model, data_loader, device, tokenizer, agg_batches=2, k = 0):
    """
    Evaluate the model with gradient caching for one peptide against all receptors.

    Args:
        model: The model to be evaluated.
        data_loader: DataLoader for the evaluation data.
        device: Device to place the data and model on.
        tokenizer: Tokenizer for encoding peptides and receptors.
        agg_batches (int): Number of batches to aggregate for evaluation.
        k (int): Index of the peptide for evaluation.

    Returns:
        list: List of counts of receptors with higher logits.
    """
    model.eval()

    cache_x = []
    cache_y = []
    big_batches = []

    onerec = None
    inc = 0
    for step, sub_batch in enumerate(data_loader):
        xx, yy = sub_batch
        if k//len(xx) == inc:
            onerec = xx[k % len(sub_batch[1])]
        inc += 1

    for step, sub_batch in enumerate(data_loader):
        xx, yy = sub_batch
        xx = (onerec,) * len(xx)

        xx = tokenizer(xx, return_tensors='pt', padding=True).to(device)
        yy = tokenizer(yy, return_tensors='pt', padding=True).to(device)
        xx['temperature'] = model.temperature
        yy['temperature'] = model.temperature

        rx, cx = _call_model_gc(model.pep_encoder, xx)
        ry, cy = _call_model_gc(model.rec_encoder, yy)

        cache_x.append(rx)
        cache_y.append(ry)
        if (step + 1) % agg_batches == 0:
            logits = get_logits(cache_x, cache_y)
            correct_logit = torch.diag(logits).tolist()[k]
            # big_batches.append(np.argsort(torch.diag(logits).tolist())[k])

            greater_logits_count = torch.sum(torch.diag(logits) > correct_logit).item() + 1
            big_batches.append(greater_logits_count)

            cache_x = []
            cache_y = []

            return big_batches

        

@cached
@autocast()
def _call_model_gc(model, input):
    """
    Cache and apply the model with gradient caching.

    Args:
        model: The model to be called.
        input: Input data to be processed by the model.

    Returns:
        tuple: A tuple containing model output and closure function.
    """
    return model(input)

@cat_input_tensor
@autocast()
def _contrastive_loss_gc(x, y):
    """
    Compute the contrastive loss for gradient-cached embeddings.

    Args:
        x: Cached embeddings for the first modality.
        y: Cached embeddings for the second modality.

    Returns:
        torch.Tensor: Contrastive loss.
    """
    logits = torch.matmul(x, y.transpose(0, 1))
    exp_logits = torch.exp(logits)
    L_r = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=1)))
    L_p = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=0)))
    loss = (L_r + L_p) * 0.5
    return loss