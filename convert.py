import music21 as m21
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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
for i in range(50):
    sequence, label = dataset[i]
    print(f"Sample {i+1}:")
    print(f"Sequence: {sequence.tolist()}")
    print(f"Label: {label.item()}")
    print()
print(f"Dataset length: {len(dataset)}")