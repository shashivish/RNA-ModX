'''
Helper Function for Feature Encoding.
'''

import pickle
import torch
import numpy as np
import torch.nn as nn
import json
import xgboost as xgb

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


class RNAClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, output_dim):
        super(RNAClassifier, self).__init__()

        # Pytroch Embedding
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # LSTM Model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.long()

        # Added Additional Embedding Layer
        x = self.embedding(x)

        # h0 and c0 are used to initialize the lstm model
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = out.squeeze(1)
        return out


class RNAPredictor():
    def __init__(self, encoder_file_path, model_directory_path):
        # self.rna_transformer = RNATransformerModel()
        # self.rna_lstm = RNAClassifier()
        print("encoder_file_path: ", encoder_file_path)
        print("model_directory_path: ", model_directory_path)
        self.encoder_file_path = encoder_file_path
        self.model_directory_path = model_directory_path

    def get_position_class_mapping(self, target):
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

    def encode_with_k_mer_codon(self, rna_sequence, kmer_dict, k):
        encoded_sequence = []
        for i in range(len(rna_sequence) - k + 1):
            encoded_sequence.append(kmer_dict[rna_sequence[i:i + k]])
        return np.array(encoded_sequence)

    def encode_sequence(self, rna_sequence: str, encoding_file_path: str):
        k = 3
        kmer_dict = {}
        kmer_dict = self.get_encoder_dictionary(encoding_file_path)

        #     print(f"Encoding file successfully loaded.")

        if len(rna_sequence) != 101:
            raise ValueError('Invalid Sequence Length. Expected Sequence Length is 101.')

        x_encoded = self.encode_with_k_mer_codon(rna_sequence, kmer_dict, k)
        X_encoded = torch.tensor([x_encoded], dtype=torch.float32)

        return X_encoded

    def get_encoder_dictionary(self, encoding_file_path):
        try:
            with open(encoding_file_path, 'rb') as f:
                kmer_dict = pickle.load(f)
        except FileNotFoundError:
            raise ValueError("File not found! Please ensure the file path is correct: " + encoding_file_path)
        except Exception as e:
            raise ValueError("An error occurred while loading the file: " + str(e))
        return kmer_dict

    def get_overall_binary_encode(self, overall_binary_encoder_path):
        try:
            with open(overall_binary_encoder_path, 'rb') as f:
                kmer_dict = pickle.load(f)
        except FileNotFoundError:
            raise ValueError("File not found! Please ensure the file path is correct.")
        except Exception as e:
            raise ValueError("An error occurred while loading the file: " + str(e))
        return kmer_dict

    """
    Takes Input RNA Sequence and returns True/False is Sequence is Modified or not.
    """

    def is_modified_nucleoside(self, rna_sequence):
        k = 3
        model_path = self.model_directory_path + "/" + "XGB_OverallBinary.pkl"
        with open(model_path, 'rb') as file:
            xgb_model = pickle.load(file)

        kmer_dict = self.get_overall_binary_encode(self.encoder_file_path)
        x_encoded = self.encode_with_k_mer_codon(rna_sequence, kmer_dict, k)

        predictions = xgb_model.predict([x_encoded])
        print("XGB Prediction ", predictions)
        if predictions[0] == 1:
            return True
        else:
            return False

    '''
    sequence : input 101 sequence to get middle position as target class.
    '''

    def get_middle_nucleoside(self, rna_sequence: str) -> str:
        target = rna_sequence[50]
        return target

    def get_predictions(self, sequence):
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

            if not self.is_modified_nucleoside(subseq):
                continue

            target = self.get_middle_nucleoside(subseq)

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

            x_train = self.encode_sequence(subseq, self.encoder_file_path)
            list_of_probabilities_for_each_class = []

            # Predict Binary Probabilities
            for c in prediction_class:
                data = {}
                model_path = self.model_directory_path + "/" + c + "_model.pth"

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
                multi_class_model_path = self.model_directory_path + "/multi-class-type-UT.pt"
            else:
                multi_class_model_path = self.model_directory_path + "/multi-class-type-" + target + ".pt"

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

            target_positional_mapping = self.get_position_class_mapping(target)
            list_of_multiclass_probabilities = []
            for i, value in enumerate(probabilities):
                data = {}
                key = target_positional_mapping[str(i)]
                data[key] = value.item()
                list_of_multiclass_probabilities.append(data)
            print("Multi Class", list_of_multiclass_probabilities)
            rna_index_modification_data = {"RNA_MODIFIED_INDEX": middle_position_index,
                                           "PARENT_MODIFIED_NUCLEOTIDE": target,
                                           "BINARY_MODIFICATION_PROBABILITIES": list_of_probabilities_for_each_class,
                                           "MULTICLASS_MODIFICATION_PROBABILITIES": list_of_multiclass_probabilities}

            response["POSITION_WITH_PROBABILITIES"].append(rna_index_modification_data)

            # i += 1
            middle_position_index += 1
        json_string = json.dumps(response)
        return json_string


if __name__ == '__main__':
    encoding_file = './3-mer-dictionary.pkl'
    # sequence = 'GGGAGGAGGGAGGATGCGCTGTGGGGTTGTTTTTGCCATAAGCGAACTTTGTGCCTGTCCTAGAAGTGAAAATTGTTCAGTCCAAGAAACTGATGTTATTT'
    sequence = "GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA"

    rna_predictor = RNAPredictor(encoder_file_path=encoding_file, model_directory_path="model")
    response = rna_predictor.get_predictions(sequence)
    print(response)
