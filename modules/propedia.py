from pathlib import Path
import requests

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