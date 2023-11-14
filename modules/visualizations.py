import os
import matplotlib.pyplot as plt
import torch


def plot_embedding_cosine_similarities(base_path, title, data_loader, tokenizer, model, device):
    curr_peptides, curr_receptors = next(iter(data_loader))
    curr_peptides = tokenizer(curr_peptides, return_tensors='pt', padding=True).to(device)
    curr_receptors = tokenizer(curr_receptors, return_tensors='pt', padding=True).to(device)

    similarity_matrix = _compute_embedding_cosine_similarities(model, curr_peptides, curr_receptors)
    similarity_matrix_np = similarity_matrix.cpu().detach().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(similarity_matrix_np, cmap="ocean", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Receptor Protein")
    plt.ylabel("Peptide")
    plot_path = _save_plot(base_path)
    print(f"{title} plot saved to {plot_path}")

def _compute_embedding_cosine_similarities(model, peptides, receptors):
    peptides_embedding, receptors_embedding = model(peptides, receptors)
    similarity_matrix = torch.mm(peptides_embedding, receptors_embedding.t())
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
