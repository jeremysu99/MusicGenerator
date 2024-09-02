import music21 as m21
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

class MusicDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label =self.labels[idx]
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.clone().detach().long()
        else:
            sequence = torch.tensor(sequence, dtype=torch.long)

        # Check if label is already a tensor
        if isinstance(label, torch.Tensor):
            label = label.clone().detach().long()
        else:
            label = torch.tensor(label, dtype=torch.long)
        return sequence, label

class MusicGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=2):
        super(MusicGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # Convert token indices to embeddings
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out
    
def midi_to_sequence(midi_file):
    midi = m21.converter.parse(midi_file)
    notes = []
    for element in midi.flatten().notes:
        if isinstance(element, m21.note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, m21.chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

kenFiles = [f for f in os.listdir("music/ken")]
cartiFiles = [f for f in os.listdir("music/carti")]
loneFiles = [f for f in os.listdir("music/lone")]

kenFullPath = [os.path.join("music/ken", f) for f in kenFiles]
cartiFullPath = [os.path.join("music/carti", f) for f in cartiFiles]
loneFullPath = [os.path.join("music/lone", f) for f in loneFiles]

kenSequences = [midi_to_sequence(file) for file in kenFullPath]
#kenSequences += [midi_to_sequence(file) for file in cartiFullPath]
#kenSequences = [midi_to_sequence(file) for file in loneFullPath]

unique_tokens = set(token for sequence in kenSequences for token in sequence)
token_to_index = {token: idx for idx, token in enumerate(unique_tokens)}
index_to_token = {idx: token for token, idx in token_to_index.items()}

indexed_sequences = [[token_to_index[token] for token in sequence] for sequence in kenSequences]

X = []
y = []

for sequence in indexed_sequences:
    for i in range(1, len(sequence)):
        X.append(sequence[:i])  # Input sequence (slice)
        y.append(sequence[i])    # Label (next token)
X_padded = pad_sequence([torch.tensor(seq) for seq in X], batch_first=True)
y = torch.tensor(y)

# Assuming X_padded and y are your processed sequences and labels
dataset = MusicDataset(X_padded, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

vocab_size = len(unique_tokens)
embedding_dim = 128  # Choose a suitable embedding size
hidden_size = 256
output_size = vocab_size  # Since you're predicting the next token

model = MusicGenerator(vocab_size, embedding_dim, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):  # Number of epochs
    for batch_sequences, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_sequences)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
    
torch.save(model.state_dict(), 'model.pth')

def generate_sequence(model, seed_sequence, length=100):
    model.eval()
    generated_sequence = seed_sequence[:]
    for _ in range(length):
        input_seq = torch.tensor(generated_sequence[-1:]).unsqueeze(0)
        output = model(input_seq)
        _, predicted = torch.max(output, 1)
        generated_sequence.append(predicted.item())
    return generated_sequence

def sequence_to_midi(sequence, output_path="generated_music.mid"):
    stream = m21.stream.Stream()
    for token in sequence:
        element = index_to_token[token]
        if '.' in element or element.isdigit():  # Chord
            chord_notes = element.split('.')
            notes = [m21.note.Note(int(n)) for n in chord_notes]
            stream.append(m21.chord.Chord(notes))
        else:  # Note
            stream.append(m21.note.Note(element))
    stream.write('midi', fp=output_path)
    print(f"MIDI file saved to {output_path}")

seed_sequence = [token_to_index[token] for token in ["C4", "E4", "G4"]]  # Example seed
generated_sequence = generate_sequence(model, seed_sequence, length=100)
sequence_to_midi(generated_sequence)