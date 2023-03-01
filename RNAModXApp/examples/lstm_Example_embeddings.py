import torch
import torch.nn as nn

# Define the sample RNA sequence

RNA_sequence = 'GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA'

# Encode the RNA sequence as integers
nucleotides = ['A', 'C', 'G', 'T']
encoded_sequence = [nucleotides.index(nucleotide) for nucleotide in RNA_sequence if nucleotide in nucleotides]

# Convert the encoded RNA sequence to a tensor
X_train = torch.tensor([encoded_sequence], dtype=torch.long)

# Define the number of unique nucleotides
num_nucleotides = len(nucleotides)

# Define the embedding layer
embedding = nn.Embedding(num_nucleotides, 10)

# Pass the RNA sequence through the embedding layer
X_train_embedded = embedding(X_train)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


# Initialize the model and set it to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(10, 100, 1, 4).to(device)

X_train_embedded = X_train_embedded.to(device)
y_train = torch.tensor([1], dtype=torch.long)  # Replace this with your actual target labels
y_train = y_train.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):

    optimizer.zero_grad()

    outputs = model(X_train_embedded)
    loss = criterion(outputs, y_train)

    loss.backward(retain_graph=True)

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

# Evaluate the model
with torch.no_grad():
    outputs = model(X_train_embedded)
    _, predicted = torch.max(outputs.data, 1)

    correct = (predicted == y_train).sum().item()
    accuracy = correct / y_train.size(0)
    print(f'Accuracy: {accuracy}')
