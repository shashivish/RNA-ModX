'''
Helper Function for Feature Encoding.
'''

import pickle
import torch
import numpy as np
import torch.nn as nn
import json


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

    #     print(f"Encoding file successfully loaded.")

    if len(rna_sequence) != 101:
        raise ValueError('Invalid Sequence Length. Expected Sequence Length is 101.')

    x_encoded = encode_with_k_mer_codon(rna_sequence, kmer_dict, k)
    X_encoded = torch.tensor([x_encoded], dtype=torch.long)

    return X_encoded


'''
sequence : input 101 sequence to get middle position as target class.
'''


def get_target_prediction_class_based_on_middle_position(rna_sequence: str) -> str:
    target = rna_sequence[50]
    return target


'''

'''


def get_predictions(sequence, encoding_file):
    prediction_class_mapping = {"A": ['hAm', 'hm1A', 'hm6A', 'hm6Am', 'Atol'], "G": ['hGm', 'hm7G'],
                                "C": ['hm5C', 'hCm'],
                                "T": ['hTm', 'hm5U', 'hPsi']}

    response = {"COMPLETE_RNA_SEQUENCE": sequence, "POSITION_WITH_PROBABILITIES": []}
    i = 0
    middle_position_index = 51
    while i + 101 <= len(sequence):

        subseq = sequence[i:i + 101]
        print('predict for sub sequence:', subseq)

        target = get_target_prediction_class_based_on_middle_position(subseq)

        # A , C , G , T/U
        if target == 'A':
            print("Detected Pipeline for A")
            prediction_class = prediction_class_mapping[target]
        elif target == "G":
            print("Detected Pipeline for G")
            prediction_class = prediction_class_mapping[target]
        elif target == "C":
            print("Detected Pipeline for C")
            prediction_class = prediction_class_mapping[target]
        elif target == "T" or target == "U":
            print("Detected Pipeline for T/U")
            prediction_class = prediction_class_mapping["T"]
        else:
            print("Invalid Nucleoside Detected.")
            prediction_class = []

        x_train = encode_sequence(subseq, encoding_file)
        list_of_probabilities_for_each_class = []
        for c in prediction_class:
            data = {}
            model_path = "../model/" + c + "_model.pt"

            # If you have GPU then remove map location parameter
            model = torch.load(model_path, map_location=torch.device('cpu'))

            # 0  - Non Modified RNA Nucleoside
            # 1  - Corresponding Modified Nucleoside

            model.eval()
            with torch.no_grad():
                output = model(x_train)
                #         print("Raw Output : ", output)
                probabilities = torch.sigmoid(output)
                print("Probabilities : ", probabilities)
                predicted_class = (probabilities > 0.5).float()

                data[c] = probabilities.numpy().tolist()

            list_of_probabilities_for_each_class.append(data)

        rna_index_modification_data = {"RNA_MODIFIED_INDEX": middle_position_index,
                                       "PARENT_MODIFIED_NUCLEOSIDE": target,
                                       "SUBCLASS_MODIFICATION_PROBABILITIES": list_of_probabilities_for_each_class}

        response["POSITION_WITH_PROBABILITIES"].append(rna_index_modification_data)

        i += 1
        middle_position_index += 1
    json_string = json.dumps(response)
    return json_string


if __name__ == '__main__':
    encoding_file = './3-mer-dictionary.pkl'
    # sequence = 'GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGCTGCA'
    sequence = "GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA"
    response = get_predictions(sequence, encoding_file)
    print(response)
