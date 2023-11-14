import torch 
from grad_cache.functional import cached, cat_input_tensor
from torch.cuda.amp import autocast


def train(model, data_loader, optimizer, tokenizer, device):
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        optimizer.zero_grad()
        loss = _process_batch(model, batch_data, tokenizer, device, ompute_grad=True)
        optimizer.step()
        total_loss += loss
    return total_loss / len(data_loader)

def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data in data_loader:
            loss = _process_batch(model, batch_data, tokenizer, device)
            total_loss += loss
    return total_loss / len(data_loader)

def _process_batch(model, batch_data, tokenizer, device, compute_grad=False):
    peptides, receptors = batch_data
    peptides = tokenizer(peptides, return_tensors='pt', padding=True).to(device)
    receptors = tokenizer(receptors, return_tensors='pt', padding=True).to(device)
    pep_embedding, rec_embedding = model(peptides, receptors)
    loss = _contrastive_loss(pep_embedding, rec_embedding.t())
    if compute_grad:
        loss.backward()
    return loss.item()

def _contrastive_loss(pep_embedding, rec_embedding):
    logits = torch.mm(pep_embedding, rec_embedding)
    exp_logits = torch.exp(logits)
    L_r = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=1)))
    L_p = -torch.mean(torch.log(torch.exp(torch.diag(logits)) / torch.sum(exp_logits, dim=0)))
    return (L_r + L_p) * 0.5


def train_gc(model, data_loader, tokenizer, optimizer, scaler, device, accumulated_batches=1):
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