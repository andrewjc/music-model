import pandas as pd
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MusicalNotesDataset(Dataset):
    def __init__(self, categorical, continuous, window_size=5):
        self.categorical = categorical
        self.continuous = continuous
        self.window_size = window_size

    def __len__(self):
        return len(self.categorical) - self.window_size

    def __getitem__(self, idx):

        # use a random window size between 1 and window_size
        window_size = np.random.randint(1, self.window_size + 1)

        categorical_in = self.categorical[idx: idx + window_size]
        categorical_out = self.categorical[idx + window_size]
        continuous_in = self.continuous[idx: idx + window_size]
        continuous_out = self.continuous[idx + window_size]

        # pad sequence with zeros if it is less than window_size
        if len(categorical_in) < self.window_size:
            categorical_in = np.pad(categorical_in, ((0, self.window_size - len(categorical_in)), (0, 0)))

        if len(continuous_in) < self.window_size:
            continuous_in = np.pad(continuous_in, ((0, self.window_size - len(continuous_in)), (0, 0)))

        return (Tensor(categorical_in), Tensor(continuous_in)), (Tensor(categorical_out), Tensor(continuous_out))

class StackedGRU(nn.Module):
    def __init__(self, num_continuous, num_categories, hidden_size, category_sizes, gru_layers=1):
        super(StackedGRU, self).__init__()
        self.num_continuous = num_continuous
        self.num_categories = num_categories
        self.hidden_size = hidden_size
        self.category_sizes = category_sizes
        self.gru_layers = gru_layers

        # Embedding layers for categorical variables
        embedding_output_size = 16
        self.embeddings = nn.Embedding(34, embedding_output_size)

        # Linear layer for continuous variables
        self.continuous_linear = nn.Linear(num_continuous, hidden_size)

        # GRU layer
        self.gru = nn.GRU(608, hidden_size, num_layers=gru_layers, batch_first=True, dropout=0.2, bidirectional=True)

        # Output layer for n+1 predictions
        self.head_continuous = nn.Linear(2 * hidden_size, num_continuous)
        self.head_categorical = nn.Linear(2 * hidden_size, num_categories)

        self.proj1 = nn.Linear(embedding_output_size + hidden_size, 2 * hidden_size)

        self.init_weights()



    def forward(self, x_categorical, x_continuous, sequence_lengths, hidden_state):
        batch_size, seq_len, _ = x_categorical.size()

        x_embedded = self.embeddings(x_categorical.long())

        x_embedded = x_embedded.view(batch_size, seq_len, -1)

        x_continuous = self.continuous_linear(x_continuous)

        # Combine continuous and categorical data
        x_combined = torch.cat([x_continuous, x_embedded], dim=2)

        #x_proj = self.proj1(x_combined)

        # Pack the sequence for GRU
        x_packed = nn.utils.rnn.pack_padded_sequence(x_combined, sequence_lengths, batch_first=True,
                                                     enforce_sorted=False)

        # GRU forward pass
        gru_out, hidden_state = self.gru(x_packed, hidden_state)

        # Unpack GRU output
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        # Predict n+1 output
        cat_output = self.head_categorical(F.relu(F.layer_norm(gru_out[:, -1, :], normalized_shape=gru_out[:, -1, :].shape)))
        cont_output = self.head_continuous(F.relu(F.layer_norm(gru_out[:, -1, :], normalized_shape=gru_out[:, -1, :].shape)))

        return cont_output, cat_output, hidden_state

    def init_weights(self):
        # Initialize embeddings
        nn.init.xavier_normal_(self.embeddings.weight.data)

        # Initialize linear layers
        nn.init.xavier_normal_(self.continuous_linear.weight.data)
        nn.init.zeros_(self.continuous_linear.bias.data)

        nn.init.xavier_normal_(self.head_continuous.weight.data)
        nn.init.zeros_(self.head_continuous.bias.data)

        nn.init.xavier_normal_(self.head_categorical.weight.data)
        nn.init.zeros_(self.head_categorical.bias.data)

        # Initialize GRU
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            else:
                nn.init.zeros_(param.data)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data  # Use weight of a parameter for initialization
        hidden = weight.new_zeros(self.gru_layers * 2, batch_size, self.hidden_size)
        return hidden


# Training
num_epochs = 100000
notes_df = pd.read_parquet('allNotes.parquet')
notes_df = notes_df.reset_index(drop=True)

# encode NoteName to categorical encoding
noteEncoder = LabelEncoder()
notes_df['NoteName'] = noteEncoder.fit_transform(notes_df['NoteName'])
pitchClassEncoder = LabelEncoder()
notes_df['PitchClass'] = pitchClassEncoder.fit_transform(notes_df['PitchClass'])
octaveEncoder = LabelEncoder()
notes_df['Octave'] = octaveEncoder.fit_transform(notes_df['Octave'])

# drop rows where Velocity is 0
notes_df = notes_df[notes_df['Velocity'] != 0]

# drop the offset column
notes_df = notes_df.drop(columns=['Offset']).astype('float32')

noteNameEncoded = pd.get_dummies(notes_df['NoteName'], columns=['NoteName']).values
pitchClassEncoded = pd.get_dummies(notes_df['PitchClass'], columns=['PitchClass']).values
octaveEncoded = pd.get_dummies(notes_df['Octave'], columns=['Octave']).values

noteNameEncodedOffset = noteNameEncoded.shape[1]
pitchClassEncodedOffset = pitchClassEncoded.shape[1]
octaveEncodedOffset = octaveEncoded.shape[1]

categorical_vars = np.concatenate((pitchClassEncoded, octaveEncoded, noteNameEncoded), axis=1)
continuous_vars = notes_df[['Duration', 'Velocity']].values

num_layers = 6
hidden_size = 64
batch_size = 256
seq_len = 12
features = 5

dataset = MusicalNotesDataset(categorical_vars, continuous_vars, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_continuous = 2  # Number of continuous variables
num_categories = 34  # Number of categorical variables
category_sizes = [len(pitchClassEncoder.classes_), len(octaveEncoder.classes_), len(noteEncoder.classes_)]

model = StackedGRU(num_continuous, num_categories, hidden_size, category_sizes, num_layers).to(device)

continuousLossFn = nn.HuberLoss()
categoricalLossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=2)

eval_loss = 0
continuous_loss = 0
categorical_loss = 0
best_loss = np.inf

for epoch in range(num_epochs):

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, batch in pbar:
        model.train()
        x, y = batch
        x_categorical, x_continuous = x

        y_categorical, y_continuous = y

        x_categorical = x_categorical.long().to(device)
        x_continuous = x_continuous.float().to(device)
        y_categorical = y_categorical.float().to(device)
        y_continuous = y_continuous.float().to(device)


        hidden_state = model.init_hidden(batch_size)

        # Get sequence lengths
        sequence_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

        # Forward pass
        continuous_output, categorical_outputs, hidden_state = model(x_categorical, x_continuous, sequence_lengths, hidden_state)

        # split the categorical_outputs
        pitchClass = categorical_outputs[:, 0:pitchClassEncodedOffset]
        octave = categorical_outputs[:, pitchClassEncodedOffset:pitchClassEncodedOffset + octaveEncodedOffset]
        noteName = categorical_outputs[:, pitchClassEncodedOffset + octaveEncodedOffset:]

        y_pitchClass = y_categorical[:, 0:pitchClassEncodedOffset]
        y_octave = y_categorical[:, pitchClassEncodedOffset:pitchClassEncodedOffset + octaveEncodedOffset]
        y_noteName = y_categorical[:, pitchClassEncodedOffset + octaveEncodedOffset:]

        categorical_outputs = [pitchClass, octave, noteName]
        y_categorical = [y_pitchClass, y_octave, y_noteName]


        # Loss calculation
        loss = 0
        continuous_loss = continuousLossFn(continuous_output, y_continuous)

        categorical_loss = sum(categoricalLossFn(xx, yy) for xx, yy in zip(categorical_outputs, y_categorical))

        #categorical_loss = categoricalLossFn(categorical_outputs, y_categorical)

        loss = continuous_loss + categorical_loss

        loss.backward()

        # Update weights
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()

        continuous_loss = continuous_loss.item()
        categorical_loss = categorical_loss.item()

        if idx % 1000 == 0 and loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_model_state.pth')
            torch.save(model.state_dict(), 'model_epoch_' + str(epoch) + '_loss_' + str(loss.item()) + '.pth')

        if idx > 0 and idx % 100000 == 0:
            model.eval()
            with torch.no_grad():
                # evaluate the model by predicting the next step and comparing it to the actual next step
                x, y = next(iter(eval_dataloader))
                # Initial input
                x_categorical = x[:, :, 0:3].long().to(device)
                x_continuous = x[:, :, 3:].float().to(device)
                y_categorical = y[:, 0:3].float().to(device)
                y_continuous = y[:, 3:].float().to(device)
                eval_loss = 0

                hidden_state = model.init_hidden(1)
                for timestep in range(15):
                    # Assume the sequence length is dynamically adjusted based on input size
                    sequence_lengths = torch.tensor([seq_len] * eval_dataloader.batch_size, dtype=torch.int32)

                    # Forward pass
                    continuous_output, categorical_outputs, hidden_state = model(x_categorical, x_continuous, sequence_lengths, hidden_state)

                    if timestep == 0:
                        # Loss calculation only for the first prediction against the true labels
                        loss_continuous = continuousLossFn(continuous_output, y_continuous)
                        loss_categorical = sum(
                            categoricalLossFn(categorical_outputs[:, i], y_categorical[:, i]) for i in
                            range(categorical_outputs.shape[1]))

                        eval_loss += loss_continuous + loss_categorical
                        eval_loss = eval_loss.item()

                    catOut = [categorical_outputs[:, i].squeeze(0) for i in range(categorical_outputs.shape[1])]

                    # decode the categorical outputs
                    try:
                        pitchClass = pitchClassEncoder.inverse_transform(catOut[0].cpu().unsqueeze(0))
                    except:
                        pass

                    try:
                        octave = octaveEncoder.inverse_transform(catOut[1].cpu().unsqueeze(0))
                    except:
                        pass

                    try:
                        noteName = noteEncoder.inverse_transform(catOut[2].cpu().unsqueeze(0))
                    except:
                        pass


                    predicted_categorical = categorical_outputs
                    predicted_continuous = continuous_output  # Take the last timestep continuous output

                    # Update x for the next prediction; this step is highly dependent on your data structure
                    # This is a conceptual example and may need adjustment
                    new_x_categorical = torch.cat((x_categorical, predicted_categorical.unsqueeze(1)), dim=1)[:, 1:, :]
                    new_x_continuous = torch.cat((x_continuous, predicted_continuous.unsqueeze(1)), dim=1)[:, 1:, :]

                    x_categorical, x_continuous = new_x_categorical.long(), new_x_continuous

            model.train()

        pbar.set_description(f'Epoch {epoch + 1}, Total Loss: {loss.item():.6f}, Cont Loss: {continuous_loss:.6f}, Cat Loss: {categorical_loss:.6f}, Eval Loss: {eval_loss:.6f}, Best Loss: {best_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        pbar.update()


        scheduler.step()

    if epoch % 100 == 0:
        torch.save(model.state_dict(), 'model_state.pth')
        print("Model state saved at epoch", epoch)
