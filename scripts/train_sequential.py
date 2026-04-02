import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

PROCESSED_PATH = "data/processed/games_with_features.csv"
MODEL_PATH = "models/lstm_sequence_model.pt"
SEQUENCE_LENGTH = 8
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3


def load_sequences(path=PROCESSED_PATH, sequence_length=SEQUENCE_LENGTH):
    df = pd.read_csv(path)
    if 'WIN' not in df.columns:
        raise ValueError('Processed dataset must contain WIN label.')

    df = df.dropna(subset=['WIN'])
    df['WIN'] = df['WIN'].astype(int)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

    numeric = df.select_dtypes(include=[np.number]).drop(columns=['WIN'], errors='ignore')
    numeric_columns = numeric.columns.tolist()

    X = []
    y = []
    for _, group in df.groupby('TEAM_ID'):
        values = group[numeric_columns].fillna(0).values
        labels = group['WIN'].values
        for idx in range(sequence_length, len(values)):
            X.append(values[idx-sequence_length:idx])
            y.append(labels[idx])

    if len(X) == 0:
        return None, None, numeric_columns

    X = np.stack(X)
    y = np.array(y)
    return X, y, numeric_columns


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return torch.sigmoid(output).squeeze(1)


def train_sequential():
    X, y, input_columns = load_sequences()
    if X is None:
        print('Not enough historical sequences to train a sequential model yet.')
        return

    dataset = SequenceDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(input_size=X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for seq_batch, label_batch in loader:
            optimizer.zero_grad()
            output = model(seq_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq_batch.size(0)
        print(f'Epoch {epoch+1}/{EPOCHS} loss: {total_loss / len(dataset):.4f}')

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved sequential model to {MODEL_PATH}')


def main():
    train_sequential()


if __name__ == '__main__':
    main()
