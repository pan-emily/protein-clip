import os 
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import EsmModel, EsmTokenizer

from modules import seed, models, data_utils_2protein, visualizations, training_utils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed.set_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_path = f'{os.getcwd()}/runs/{timestamp}'
    os.makedirs(base_path, exist_ok=True)
    print(f"All run info will be saved to {base_path}")

    # set pre-trained model for encoder
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D").to(device)
    for param in esm_model.parameters():
        param.requires_grad = False
    input_dim = 640

    # set model hyperparameters
    embedding_dim = 128
    h1 = 2
    h2 = 2
    dropout = 0.1
    trained_model = models.ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model).to(device)

    # set dataloader hyperparameters
    batch_size = 16
    train_dataset, val_dataset, test_dataset = data_utils_2protein.generate_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    data_dir = Path('data')
    visualizations.plot_clustering(base_path, data_dir, prefix='protein2')
    visualizations.plot_protein_lengths(base_path, data_dir)

    # set training hyperparameters 
    num_epochs = 15
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=1e-3)
    training_with_grad_cache = True
    if training_with_grad_cache:
        scaler = GradScaler()
        accumulated_batches = 1

    # init before training
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    visualizations.plot_embedding_cosine_similarities(base_path, "Raw Embedding Cosine Similarities", train_loader, tokenizer, trained_model, device)
    model_save_path = '/groups/mlprojects/protein-clip-pjt/protein-clip/runs/20231202_132632_854863/best_model.pth'
    # losses_save_path = f'{base_path}/losses_per_epoch.txt'
    print(f"Best model will be saved to {model_save_path}")
    # print(f"Losses will be saved to {losses_save_path}")
    
    # analyzing performance after training
    best_trained_model = models.ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model).to(device)
    best_trained_model.load_state_dict(torch.load(model_save_path))
    perf = []
    for i in range(256):
        curr_perf = np.mean(training_utils.eval_gc_allrec_onepep(best_trained_model, val_loader, device, tokenizer, trained_model, 
                          val_loader, agg_batches=16, k = i))
        perf.append(curr_perf)
        print(curr_perf)
        

    top_k = [0]*256
    for i in perf:
        top_k[int(i)] += 1

    for i in range(1, 256):
        top_k[i] += top_k[i-1]

    accs = []
    for i in range(256):
        accs.append(top_k[i]/256)

    numbers = np.arange(256)
    np.random.shuffle(numbers)
    perf2 = numbers.tolist()

    top_k2 = [0]*256
    for i in perf2:
        top_k2[int(i)] += 1
    for i in range(1, len(perf2)):
        top_k2[i] += top_k2[i-1]
    accs2 = []
    for i in range(256):
        accs2.append(top_k2[i]/256)

    k_vals = np.linspace(0, 256, 256)

    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, accs, label='Model Top-k accuracy', color='blue')
    plt.plot(k_vals, accs2, label='Random Top-k accuracy', color='orange')
    plt.title('Top-k Val Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Top-k')
    plt.legend()
    plt.tight_layout()

    plot_path = visualizations._save_plot(base_path)
    print(f"{'Top-k Val Accuracy'} plot saved to {plot_path}")

if __name__ == '__main__':
    main() 