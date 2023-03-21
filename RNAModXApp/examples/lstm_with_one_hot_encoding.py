import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Define a sample RNA sequence
sequence = 'GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA'


# One-hot encode the RNA sequence
def one_hot_encode(sequence):
    nucleotides = ['A', 'C', 'G', 'U']
    one_hot = []
    for nucleotide in sequence:
        hot = [0 if nucleotide != nt else 1 for nt in nucleotides]
        # For G - [0,0,1,0]
        one_hot.append(hot)
    return one_hot


one_hot_encoded_data = one_hot_encode(sequence)
encoded_sequence = torch.tensor(one_hot_encoded_data, dtype=torch.float32)
encoded_sequence = encoded_sequence.unsqueeze(0)  # add batch dimension

input_dim = 4  # one-hot encoding of RNA nucleotides
hidden_dim = 128
layer_dim = 2
output_dim = 4  # binary classification of modification presence or absence

model = LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Replace the following line with your actual training data
X_train = encoded_sequence
y_train = torch.tensor([1], dtype=torch.long)  # Replace this with your actual target labels

# Train the model using the training dataset
for epoch in range(100):
    # forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/100], Loss: {:.4f}'.format(epoch + 1, loss.item()))

# Replace the following line with your actual target labels
y_train = torch.tensor([1, 0, 0, 0], dtype=torch.long)

with torch.no_grad():
    outputs = model(X_train)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = torch.sum(predicted.eq(y_train)).item() / y_train.shape[0]

print(f'Accuracy: {accuracy:.2f}')
