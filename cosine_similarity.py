import os 
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import EsmModel, EsmTokenizer

from modules import seed, models, data_utils, data_utils_2protein, visualizations
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
    visualizations.plot_embedding_cosine_similarities(base_path, "Raw Embedding Cosine Similarities", train_loader, tokenizer, trained_model, device)
    
    model_save_path = '/groups/mlprojects/protein-clip-pjt/protein-clip/runs/20231130_163836_134376/best_model.pth'

    # analyzing performance after training
    best_trained_model = models.ExtendedCLIP(input_dim, embedding_dim, h1, h2, dropout, esm_model).to(device)
    best_trained_model.load_state_dict(torch.load(model_save_path))

    visualizations.plot_embedding_cosine_similarities(base_path, "Trained Embedding Cosine Similarities on Train Set", train_loader, tokenizer, best_trained_model, device)    
    visualizations.plot_embedding_cosine_similarities(base_path, "Trained Embedding Cosine Similarities on Val Set", val_loader, tokenizer, best_trained_model, device)


if __name__ == '__main__':
    main() 