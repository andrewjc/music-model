import joblib
import pandas as pd
import numpy as np
import os

from adaml import AdamL

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from music_model import MusicPredictionModel

# Training
num_epochs = 100000
rebuild_cache = False
num_layers = 3
hidden_size = 64
batch_size = 512
seq_len = 12
features = 5
embedding_dim = 4
isSchedulerMetricBased = False

def calculate_tension(raw_note_buffer):
    """
    Calculates a scalar value representing the musical tension within a series of notes.

    Args:
        raw_note_buffer: A list of lists, where each inner list represents a note:
            [PitchClass, Octave, NoteName, Duration, Velocity, Offset]

    Returns:
        A float value between 0 and 1, where 0 represents low tension and 1 represents high tension.
    """

    # Preprocessing
    notes = np.array(raw_note_buffer)
    pitches = notes[:, 0].astype(int)  # Extract pitch classes

    # 1. Dissonance Calculation
    dissonance_scores = _calculate_dissonance(pitches)

    # 2. Harmonic Instability
    instability_scores = _calculate_instability(pitches)

    # 3. Rhythmic Complexity
    complexity_scores = _calculate_rhythmic_complexity(notes)

    # 4. Dynamic Variation
    dynamic_scores = _calculate_dynamic_variation(notes)

    # 5. Combine factors (weighted average)
    tension = 0.4 * np.mean(dissonance_scores) + \
              0.25 * np.mean(instability_scores) + \
              0.2 * np.mean(complexity_scores) + \
              0.15 * np.mean(dynamic_scores)

    return tension


def _calculate_dissonance(pitches):
    """Calculates the dissonance of intervals between consecutive notes."""
    dissonance_map = {1: 0.9, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.1, 6: 0.4, 7: 0.9}
    dissonance_scores = []
    for i in range(len(pitches) - 1):
        interval = abs(pitches[i + 1] - pitches[i]) % 12  # Interval within an octave
        dissonance_scores.append(dissonance_map.get(interval, 0))  # Default to 0 if interval not found
    return dissonance_scores


def _calculate_instability(pitches):
    """Measures tension based on distance from tonal centers (stability)."""
    # Simplified model: assuming C major
    stable_notes = [0, 2, 4, 5, 7, 9, 11]  # Pitch classes of C major scale
    instability_scores = []
    for pitch in pitches:
        distance = min((pitch - note) % 12 for note in stable_notes)
        # Map distance to instability (example, could be refined)
        instability = (1 / (1 + np.exp(-distance + 3)))
        instability_scores.append(instability)
    return instability_scores


def _calculate_rhythmic_complexity(notes):
    """Scores tension based on variations in note durations."""
    durations = notes[:, 3].astype(float)
    duration_diffs = np.abs(np.diff(durations))
    return np.clip(duration_diffs, a_min=0, a_max=1)  # Normalize values


def _calculate_dynamic_variation(notes):
    """Scores tension based on changes in note velocity (dynamics)."""
    velocities = notes[:, 4].astype(int)
    velocity_diffs = np.abs(np.diff(velocities)) / 127.0  # Normalize by max velocity
    return velocity_diffs

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




if rebuild_cache:
    notes_df = pd.read_parquet('allNotes.parquet')
    notes_df = notes_df.reset_index(drop=True)

    # drop rows where Velocity is 0
    notes_df = notes_df[notes_df['Velocity'] != 0]

    # set PitchClass to 0 where Velocity is 0
    notes_df.loc[notes_df['Velocity'] == 0, 'PitchClass'] = 0

    # calculate the tension in a future looking window of 12 notes
    # add a new column to the dataframe that contains the avg tension for the next 12 notes
    tension = []
    for i in range(len(notes_df)):
        if i + 12 < len(notes_df):
            tension.append(calculate_tension(notes_df.iloc[i:i+12].values))
        else:
            tension.append(0)

    notes_df['Tension'] = tension

    # encode NoteName to categorical encoding
    noteEncoder = LabelEncoder()
    notes_df['NoteName'] = noteEncoder.fit_transform(notes_df['NoteName'])
    #pitchClassEncoder = LabelEncoder()
    #notes_df['PitchClass'] = pitchClassEncoder.fit_transform(notes_df['PitchClass'])
    octaveEncoder = LabelEncoder()
    notes_df['Octave'] = octaveEncoder.fit_transform(notes_df['Octave'])

    joblib.dump(noteEncoder, 'config/noteEncoder.pkl')
    #joblib.dump(pitchClassEncoder, 'pitchClassEncoder.pkl')
    joblib.dump(octaveEncoder, 'config/octaveEncoder.pkl')

    # drop the offset column
    notes_df = notes_df.drop(columns=['PitchClass', 'Offset']).astype('float32')

    noteNameOneHotEncoder = OneHotEncoder(sparse_output=False)
    noteNameEncoded = noteNameOneHotEncoder.fit_transform(notes_df['NoteName'].values.reshape(-1, 1))
    #pitchClassOneHotEncoder = OneHotEncoder()
    #pitchClassEncoded = pitchClassOneHotEncoder.fit_transform(notes_df['PitchClass'].values.reshape(-1, 1)).toarray()
    octaveOneHotEncoder = OneHotEncoder(sparse_output=False)
    octaveEncoded = octaveOneHotEncoder.fit_transform(notes_df['Octave'].values.reshape(-1, 1))

    categorical_vars = np.concatenate((octaveEncoded, noteNameEncoded), axis=1)

    joblib.dump(noteNameOneHotEncoder, 'config/noteNameOneHotEncoder.pkl')
    #joblib.dump(pitchClassOneHotEncoder, 'pitchClassOneHotEncoder.pkl')
    joblib.dump(octaveOneHotEncoder, 'config/octaveOneHotEncoder.pkl')

    noteNameEncodedOffset = noteNameEncoded.shape[1]
    #pitchClassEncodedOffset = pitchClassEncoded.shape[1]
    octaveEncodedOffset = octaveEncoded.shape[1]

    continuous_vars = notes_df[['Duration', 'Velocity', 'Tension']].values

    # Normalize continuous variables
    scaler = StandardScaler()
    continuous_vars = scaler.fit_transform(continuous_vars)

    minMax = MinMaxScaler()
    continuous_vars = minMax.fit_transform(continuous_vars)

    joblib.dump(minMax, 'config/minMax.pkl')
    joblib.dump(scaler, 'config/scaler.pkl')

    joblib.dump(categorical_vars, 'data/categorical_vars.pkl')
    joblib.dump(continuous_vars, 'data/continuous_vars.pkl')
    joblib.dump((noteNameEncodedOffset, octaveEncodedOffset), 'config/offsets.pkl')
else:
    categorical_vars = joblib.load('data/categorical_vars.pkl')
    continuous_vars = joblib.load('data/continuous_vars.pkl')
    noteNameEncodedOffset, octaveEncodedOffset = joblib.load('config_notes/offsets.pkl')
    octaveEncoder = joblib.load('config_notes/octaveEncoder.pkl')
    noteEncoder = joblib.load('config_notes/noteEncoder.pkl')


dataset = MusicalNotesDataset(categorical_vars, continuous_vars, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_continuous = 3  # Number of continuous variables
offsets = (octaveEncodedOffset, noteNameEncodedOffset)  # Number of categorical variables
category_sizes = [len(octaveEncoder.classes_), len(noteEncoder.classes_)]

model = MusicPredictionModel(num_continuous, offsets, hidden_size, num_layers, embedding_dim, device).to(device)

print(model)

continuousLossFn = nn.HuberLoss()
categoricalLossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10000, factor=0.9, verbose=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001)

eval_loss = 0
continuous_loss = 0
categorical_loss = 0
best_loss = np.inf

# Get sequence lengths
sequence_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.int32)


def repetition_penalty(categorical_outputs, x_categorical, penalty_scale):
    """
    Penalizes the model for repeating the same note multiple times in a row.
    """

    # make a comparable note tensor
    cat_compare2_octave = categorical_outputs[0]
    cat_compare2_notes = categorical_outputs[1]

    cat_compare2_octave = F.softmax(cat_compare2_octave, dim=-1)
    cat_compare2_notes = F.softmax(cat_compare2_notes, dim=-1)
    _, predicted_octave = torch.max(cat_compare2_octave, dim=-1)
    _, predicted_note = torch.max(cat_compare2_notes, dim=-1)

    # Calculate the penalty for each note
    penalty = torch.zeros((x_categorical.shape[0]), dtype=torch.float32, device=device)
    for i in range(x_categorical[0].shape[0]):
        cat_compare_to = x_categorical[:, i, :]
        cat_compare_octave = torch.max(cat_compare_to[:, 0:octaveEncodedOffset], dim=-1).indices
        cat_compare_noteName = torch.max(cat_compare_to[:, octaveEncodedOffset:], dim=-1).indices

        compare_mask = predicted_octave == cat_compare_octave
        compare_mask = compare_mask & (predicted_note == cat_compare_noteName)

        relative_position = min(i / (x_categorical[0].shape[0] - 1), 1 - i / (x_categorical[0].shape[0] - 1))

        distance_penalty = penalty_scale * relative_position
        penalty += compare_mask.float() * distance_penalty

    return penalty


for epoch in range(num_epochs):
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, batch in pbar:
        x, y = batch
        x_categorical, x_continuous = x

        y_categorical, y_continuous = y

        x_categorical = x_categorical.long().to(device)
        x_continuous = x_continuous.float().to(device)
        y_categorical = y_categorical.float().to(device)
        y_continuous = y_continuous.float().to(device)

        hidden_state = model.init_hidden(batch_size)


        # Forward pass
        categorical_outputs, continuous_output, hidden_state = model(x_categorical, x_continuous, sequence_lengths, hidden_state)

        # split the categorical_outputs
        octave = categorical_outputs[:, 0:octaveEncodedOffset]
        noteName = categorical_outputs[:, octaveEncodedOffset:]

        y_octave = y_categorical[:, 0:octaveEncodedOffset]
        y_noteName = y_categorical[:, octaveEncodedOffset:]

        categorical_outputs = [octave, noteName]
        y_categorical = [y_octave, y_noteName]

        # Loss calculation
        loss = 0
        continuous_loss = continuousLossFn(continuous_output, y_continuous)

        categorical_loss = sum(categoricalLossFn(xx, yy) for xx, yy in zip(categorical_outputs, y_categorical))

        loss = continuous_loss + categorical_loss

        rep_penalty = repetition_penalty(categorical_outputs, x_categorical, penalty_scale=1.5)
        loss += rep_penalty.mean()  # Apply the summed penalty to the loss

        pbar.set_description(f'Epoch {epoch + 1}, Total Loss: {loss.item():.6f}, Repet Loss: {rep_penalty.mean()}, Cont Loss: {continuous_loss:.6f}, Cat Loss: {categorical_loss:.6f}, Eval Loss: {eval_loss:.6f}, Best Loss: {best_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        loss.backward()

        # Update weights
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        continuous_loss = continuous_loss.item()
        categorical_loss = categorical_loss.item()

        if idx % 1000 == 0 and loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'note_model/gru_best_model_state.pth')
            torch.save(model.state_dict(), 'gru_model_epoch_' + str(epoch) + '_idx_' + str(idx) + '_loss_' + str(loss.item()) + '.pth')
            print('Saved checkpoint at loss ' + str(loss.item()))

        pbar.update()

        if isSchedulerMetricBased:
            scheduler.step(loss)
        else:
            scheduler.step()

