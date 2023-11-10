from pathlib import Path
import requests

from Bio import SeqIO

def get_data_from_fasta(peptide_file_path, receptor_file_path):
    peptides, receptors = [], []
    
    # Read receptor sequences from a fasta file
    with open(receptor_file_path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            receptors.append(str(record.seq))

    # Read peptide sequences from a fasta file
    with open(peptide_file_path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            peptides.append(str(record.seq))

    assert len(peptides) == len(receptors), "The number of peptides and receptors must be the same"

    print(f"Imported {len(peptides)} peptides and {len(receptors)} receptors from fasta files.")

    return peptides, receptors


def get_data():
    peptide_url = 'http://bioinfo.dcc.ufmg.br/propedia/public/download/peptide.fasta'
    receptor_url = 'http://bioinfo.dcc.ufmg.br/propedia/public/download/receptor.fasta'
    peptide_data = requests.get(peptide_url).text
    receptor_data = requests.get(receptor_url).text

    with open('peptide.fasta', 'w') as f:
        f.write(peptide_data)
    with open('receptor.fasta', 'w') as f:
        f.write(receptor_data)

    peptides, receptors = [], []
    with open('receptor.fasta', 'r') as f:
        for line in f.readlines():
            if not line.startswith('>'):
                receptors.append(line.replace('\n', ''))

    with open('peptide.fasta', 'r') as f:
        for line in f.readlines():
            if not line.startswith('>'):
                peptides.append(line.replace('\n', ''))
    assert len(peptides) == len(receptors)

    print(f"Imported {len(peptides)} peptides and {len(receptors)} receptors from propedia.")

    return peptides, receptors 