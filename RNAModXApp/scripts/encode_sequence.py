'''
Helper Function for Feature Encoding.
'''

import pickle
import torch
import numpy as np
import torch.nn as nn


class RNATransformerModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, output_dim, dropout=0.5):
        super(RNATransformerModel, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # If batch size first is true then it should be batch size , sequence lenght , embedding dimension
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=hidden_dim,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.long()
        # print("Shape of Original X  ", x.shape)
        x_embedded = self.embedding(x)
        # print("Shape of X embedded" , x_embedded.shape)
        x_transformed = self.transformer_encoder(x_embedded)
        # print("Shape of Transformed X" , x_transformed.shape)
        x_transformed = x_transformed[:, -1, :]  # taking the last token's output

        output = self.dropout(x_transformed)
        out = self.fc(output)
        return out.squeeze()


def encode_with_k_mer_codon(rna_sequence, kmer_dict, k):
    encoded_sequence = []
    for i in range(len(rna_sequence) - k + 1):
        encoded_sequence.append(kmer_dict[rna_sequence[i:i + k]])
    return np.array(encoded_sequence)


def encode_sequence(rna_sequence: str, encoding_file_path: str):
    k = 3
    kmer_dict = {}
    try:
        with open(encoding_file_path, 'rb') as f:
            kmer_dict = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("File not found! Please ensure the file path is correct.")
    except Exception as e:
        raise ValueError("An error occurred while loading the file: " + str(e))

    print(f"Encoding file successfully loaded.")

    if len(rna_sequence) != 101:
        raise ValueError('Invalid Sequence Length. Expected Sequence Length is 101.')

    x_encoded = encode_with_k_mer_codon(rna_sequence, kmer_dict, k)
    X_encoded = torch.tensor([x_encoded], dtype=torch.long)

    return X_encoded


if __name__ == '__main__':
    encoding_file = 'C:/Users/shashi.vish/Documents/Shashi/Education/HigherEducation/NUS/Capstone Project/Git/RNA-ModX/RNAModXApp//notebooks/model_building/LSTM//3-mer-dictionary.pkl'
    sequence = 'GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGG'
    x_train = encode_sequence(sequence, encoding_file)
    print(x_train)

    model_path = "../../../models/Transfomer_3Mer/hAm_model.pt"

    # If you have GPU then remove map location parameter
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print(model)

    print("Shape of X_train ", x_train.shape)

    # 0  - Non Modified RNA Nucleoside
    # 1  - Corresponding Modified Nucleoside

    model.eval()
    with torch.no_grad():
        output = model(x_train)
        print("Raw Output : ", output)
        probabilities = torch.sigmoid(output)
        print("Probabilities : ", probabilities)
        predicted_class = (probabilities > 0.5).float()
        print("Predicted Class : ", predicted_class)
