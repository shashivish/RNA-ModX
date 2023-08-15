'''
Helper Function for Feature Encoding.
'''

import pickle
import torch
import numpy as np
import torch.nn as nn
import json

from torch.nn import DataParallel


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


def get_position_class_mapping(target):
    # "hm5C -  0
    # hCm - 1"

    # "hTm  - 0
    # hm5U - 1
    # hPsi - 2"

    # "hGm  - 0
    # hm7G - 1"

    # "hm6A - 0
    # hm1A - 1
    # hAm  - 2
    # Atol - 3
    # hm6Am -4"

    if target == "A":
        return {"0": "hm6A", "1": "hm1A", "2": "hAm", "3": "Atol", "4": "hm6Am"}
    elif target == "G":
        return {"0": "hGm", "1": "hm7G"}
    elif target == "C":
        return {"0": "hm5C", "1": "hCm"}
    elif target == "T" or target == "U":
        return {"0": "hTm", "1": "hm5U", "2": "hPsi"}
    return ""


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
    X_encoded = torch.tensor([x_encoded], dtype=torch.float32)

    return X_encoded


def is_modified_nucleoside(rna_sequence):


    return True

'''
sequence : input 101 sequence to get middle position as target class.
'''


def get_middle_nucleoside(rna_sequence: str) -> str:
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
    sequence_len = 101
    middle_position_index = 51
    for i in range(len(sequence) - sequence_len + 1):

        subseq = sequence[i:i + sequence_len]
        print('predict for sub sequence:', subseq)

        if not is_modified_nucleoside(subseq):
            continue

        target = get_middle_nucleoside(subseq)

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

        # Predict Binary Probabilities
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
                probabilities = torch.sigmoid(output)
                data[c] = probabilities.numpy().tolist()

            list_of_probabilities_for_each_class.append(data)

        #

        if target == "T" or target == "U":
            multi_class_model_path = "../model/multi-class-type-UT.pt"
        else:
            multi_class_model_path = "../model/multi-class-type-" + target + ".pt"

        multi_class_model = torch.load(multi_class_model_path, map_location=torch.device('cpu'))
        device = torch.device("cpu")  # Set the device to CPU

        # Move the model to CPU
        multi_class_model = multi_class_model.to(device)
        if isinstance(multi_class_model, torch.nn.DataParallel):
            multi_class_model = multi_class_model.module.to(device)
        multi_class_model.eval()
        probabilities = []
        with torch.no_grad():
            output = multi_class_model(x_train.to(device))
            # print(output)
            probabilities = torch.softmax(output, dim=0)

        target_positional_mapping = get_position_class_mapping(target)
        list_of_multiclass_probabilities = []
        for i, value in enumerate(probabilities):
            data = {}
            key = target_positional_mapping[str(i)]
            data[key] = value.item()
            list_of_multiclass_probabilities.append(data)
        print("Multi Class", list_of_multiclass_probabilities)
        rna_index_modification_data = {"RNA_MODIFIED_INDEX": middle_position_index,
                                       "PARENT_MODIFIED_NUCLEOSIDE": target,
                                       "BINARY_MODIFICATION_PROBABILITIES": list_of_probabilities_for_each_class,
                                       "MULTICLASS_MODIFICATION_PROBABILITIES": list_of_multiclass_probabilities}

        response["POSITION_WITH_PROBABILITIES"].append(rna_index_modification_data)

        #i += 1
        middle_position_index += 1
    json_string = json.dumps(response)
    return json_string


if __name__ == '__main__':
    encoding_file = './3-mer-dictionary.pkl'
    sequence = 'GGGAGGAGGGAGGATGCGCTGTGGGGTTGTTTTTGCCATAAGCGAACTTTGTGCCTGTCCTAGAAGTGAAAATTGTTCAGTCCAAGAAACTGATGTTATTT'
    #sequence = "GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA"
    response = get_predictions(sequence, encoding_file)
    print(response)
