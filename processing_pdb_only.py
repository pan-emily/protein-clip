import os
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from modules import data_utils_2protein, visualizations

def protein_to_protein_dataset():
    # generate where figures will be saved to
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_path = f'{os.getcwd()}/runs/{timestamp}'
    os.makedirs(base_path, exist_ok=True)
    print(f"All run info will be saved to {base_path}")

    batch_size = 8

    # Generate datasets, download all proteins made of 2 protein chains from pdb
    train_dataset, val_dataset, test_dataset = data_utils_2protein.generate_datasets()

    data_dir = Path('data')
    visualizations.plot_clustering(base_path, data_dir, prefix='protein2')
    visualizations.plot_protein_lengths(base_path, data_dir)

def ppiref_dataset():
    data_utils_2protein._from_ppiref_get_or_download_data()


if __name__ == "__main__":
    ppiref_dataset()
