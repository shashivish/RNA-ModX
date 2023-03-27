import numpy as np
import xgboost as xgb


class ModelPredictionHelperClass:

    def __init__(self):
        self.MAX_LENGTH = 0
        self.WINDOW_SIZE = 0

    # Encode using one-hot encoding
    def apply_one_hot_encode(self, sequence):
        nucleotides = ['C', 'A', 'T', 'G']
        one_hot = []
        for nucleotide in sequence:
            # For G - [0,0,1,0]
            if nucleotide == 'N':  # N is not application , used for empty values.
                hot = [0, 0, 0, 0]
            else:
                hot = [0 if nucleotide != nt else 1 for nt in nucleotides]

            one_hot.append(hot)
        return np.array(one_hot).flatten()

    # Apply ANF Encoding on Input Features
    def apply_accumulated_nucle_frequency(self, seq):
        mapping = []
        A = 0
        C = 0
        T = 0
        G = 0
        for i, v in enumerate(seq):
            if v == 'A':
                A += 1
                mapping.append(A / (i + 1))
            elif v == 'C':
                C += 1
                mapping.append(C / (i + 1))
            elif v == 'T' or v == 'U':
                T += 1
                mapping.append(T / (i + 1))
            else:
                G += 1
                mapping.append(G / (i + 1))
        padding = (self.MAX_LENGTH - len(mapping))
        mapping = np.pad(mapping, (0, padding), 'constant')
        return mapping

    '''
        Perform Prediction based on Input Sequence and Window Size 
        
        return List of Position to be highlighted  in Sequence 
    
    '''

    def perform_prediction_based_on_window(self, input_sequence: str, model, window_size: int):
        list_of_position_to_be_highlighted = []
        for start in range(len(input_sequence)):
            sub_sequence_for_prediction = input_sequence[start:start + window_size + 1]
            if not len(sub_sequence_for_prediction) < window_size + 1:  # Exceeded Sequence Boundary
                print("SubSequence : ", sub_sequence_for_prediction)
                print("Length of SubSequence : ", len(sub_sequence_for_prediction))
                x_test = self.apply_accumulated_nucle_frequency(sub_sequence_for_prediction)
                print("Input Sequence Encoded : ", x_test)
                xg_matrix = xgb.DMatrix([x_test])
                y_pred = model.predict(xg_matrix)
                print("Predicted Class  : ", y_pred)
                if y_pred != 13:  # Sequence if modified if class is not 13
                    position = start + (window_size // 2)
                    list_of_position_to_be_highlighted.append(position)

        return list_of_position_to_be_highlighted

    def predict_rna_modification_status(self, rna_sequence, model, window_size):
        self.MAX_LENGTH = int(window_size) + 1
        window_size = int(window_size)
        modified_positions = self.perform_prediction_based_on_window(rna_sequence, model, window_size)
        return modified_positions
