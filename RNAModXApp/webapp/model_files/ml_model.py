import numpy as np


# Encode using one-hot encoding
def apply_one_hot_encode(sequence):
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


def predict_rna_modification_status(rna_sequence, model):
    x = apply_one_hot_encode(rna_sequence)
    x_test= []
    x_test.append(np.array(x))
    print("Predicting Sequence : ", x_test , " with Shape ", x_test[0].shape)
    y_pred = model.predict(x_test)

    print("Predicted Result : ", y_pred)
    return y_pred
