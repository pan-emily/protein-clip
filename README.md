# CLIP Model for Peptide-Receptor Interactions
CS 101 Project 
Contributors: Ayush Varshney, Emily Pan, Evan Zhang, Jadelynn Dao

# Documentation

#### TL;DR: 

Running ```main.py``` extracts proten-receptor data from Propedia, then builds + trains a CLIP model. 

Model versioning is based on time of run, so model/losses will be saved to directory determined by H-M-S we begin training the model. 


## Environment Setup

### If running locally:

Create a new Anaconda environment from ```environment.yml``` using the command
    > ```conda env create -f environment.yml```

Running scripts should simply require you to run the specified file using the anaconda version of python3. 

### If running on HPC: 

Create a new Anaconda environment from ```environment_hpc.yml``` using the command
    > ```conda env create -f environment_hpc.yml```

To run the scripts, follow the documentation for your specific computing cluster. 


## Peptide-Receptor CLIP

The peptide-receptor CLIP model should be ran using the script ```peptide-protein.py```. 

## Protein-Protein CLIP 

## Protein-Receptor FILIP 
