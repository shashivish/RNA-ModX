'''
Helper Function for Feature Encoding.
'''

import pickle
import torch
import numpy as np


def encode_with_k_mer_codon(sequence, kmer_dict, k):
    encoded_sequence = []
    for i in range(len(sequence) - k + 1):
        encoded_sequence.append(kmer_dict[sequence[i:i + k]])
    return np.array(encoded_sequence)


def encode_sequence(sequence: str, encoding_file: str):
    k = 3
    kmer_dict = {}
    try:
        with open(encoding_file, 'rb') as f:
            kmer_dict = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("File not found! Please ensure the file path is correct.")
    except Exception as e:
        raise ValueError("An error occurred while loading the file: " + str(e))

    print(f"Encoding file successfully loaded.")

    if len(sequence) != 101:
        raise ValueError('Invalid Sequence Length. Expected Sequence Length is 101.')

    x_encoded = encode_with_k_mer_codon(sequence, kmer_dict, k)
    X_encoded = torch.tensor(x_encoded, dtype=torch.long)

    return X_encoded


if __name__ == '__main__':
    encoding_file = 'C:/Users/shashi.vish/Documents/Shashi/Education/HigherEducation/NUS/Capstone Project/Git/RNA-ModX/RNAModXApp//notebooks/model_building/LSTM//3-mer-dictionary.pkl'
    sequence = 'GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGG'
    x_train = encode_sequence(sequence, encoding_file)
    print(x_train)
