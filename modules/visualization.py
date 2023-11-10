import torch 
import matplotlib.pyplot as plt 

def compute_similarity_matrix(model, peptides, receptors):
    pep_norm_embedding, rec_norm_embedding = model(peptides, receptors)
    similarity_matrix = torch.mm(pep_norm_embedding, rec_norm_embedding.t())
    return similarity_matrix * torch.exp(-model.t)

def plot_raw_embedding_cosine_similarities(train_loader, tokenizer, trained_model, device):
    curr_peptides, curr_receptors = next(iter(train_loader))
    curr_peptides = tokenizer(curr_peptides, return_tensors='pt', padding=True).to(device)
    curr_receptors = tokenizer(curr_receptors, return_tensors='pt', padding=True).to(device)

    similarity_matrix = compute_similarity_matrix(trained_model, curr_peptides, curr_receptors)
    similarity_matrix_np = similarity_matrix.cpu().detach().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(similarity_matrix_np, cmap="ocean", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Raw Embedding Cosine Similarities")
    plt.xlabel("Receptor Protein")
    plt.ylabel("Peptide")
    plt.show()

    plt.savefig('/output/raw_embedding_cosine_similarities.png')

    del curr_peptides
    del curr_receptors
    del similarity_matrix
    del similarity_matrix_np