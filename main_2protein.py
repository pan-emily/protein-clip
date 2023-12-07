import os 
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import EsmModel, EsmTokenizer

from modules import seed, models, data_utils_2protein, visualizations, training_utils
from pathlib import Path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed.set_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp = "20231201_190729_089146"
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
    num_epochs = 10
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=1e-3)
    training_with_grad_cache = True
    if training_with_grad_cache:
        scaler = GradScaler()
        accumulated_batches = 32

    # init before training
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    visualizations.plot_embedding_cosine_similarities(base_path, "Raw Embedding Cosine Similarities", train_loader, tokenizer, trained_model, device)
    model_save_path = f'{base_path}/best_model.pth'
    losses_save_path = f'{base_path}/losses_per_epoch.txt'
    print(f"Best model will be saved to {model_save_path}")
    print(f"Losses will be saved to {losses_save_path}")

    trained_model.load_state_dict(torch.load(model_save_path))

    # training 
    with open(losses_save_path, 'w') as f:
        f.write('Epoch,Train Loss,Validation Loss\n')

        for epoch in range(num_epochs):
            print('new epoch')
            if training_with_grad_cache:
                train_loss = training_utils.train_gc(trained_model, train_loader, tokenizer, optimizer, scaler, device, accumulated_batches)
            else:
                train_loss = training_utils.train(trained_model, train_loader, optimizer, tokenizer, device)
            val_loss = training_utils.evaluate(trained_model, val_loader, tokenizer, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            f.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f}\n")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = trained_model.state_dict()
                torch.save(best_model_state, model_save_path)
                best_trained_model = models.ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model).to(device)
                best_trained_model.load_state_dict(torch.load(model_save_path))

                visualizations.plot_embedding_cosine_similarities(base_path, f"Trained Embedding Cosine Similarities on Train Set - Epoch {epoch + 1}", train_loader, tokenizer, best_trained_model, device)    
                visualizations.plot_embedding_cosine_similarities(base_path, f"Trained Embedding Cosine Similarities on Val Set - Epoch {epoch + 1}", val_loader, tokenizer, best_trained_model, device)


            torch.cuda.empty_cache()
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # analyzing performance after training
    best_trained_model = models.ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model).to(device)
    best_trained_model.load_state_dict(torch.load(model_save_path))
    test_loss = training_utils.evaluate(best_trained_model, test_loader, tokenizer, device)
    print(f"Test Loss: {test_loss:.4f}") 

    if training_with_grad_cache:
        visualizations.plot_loss_curves(base_path, train_losses, val_losses, batch_size*accumulated_batches, batch_size)
    else:
        visualizations.plot_loss_curves(base_path, train_losses, val_losses, batch_size, batch_size)

    visualizations.plot_embedding_cosine_similarities(base_path, "Trained Embedding Cosine Similarities on Train Set", train_loader, tokenizer, best_trained_model, device)    
    visualizations.plot_embedding_cosine_similarities(base_path, "Trained Embedding Cosine Similarities on Val Set", val_loader, tokenizer, best_trained_model, device)


if __name__ == '__main__':
    main() 