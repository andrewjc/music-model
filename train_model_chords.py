import math
import random

import joblib
import pandas as pd
import numpy as np
import os

from adaml import AdamL

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from music_chord_model import MusicPredictionChordModel

# Training
num_epochs = 100000
rebuild_cache = False
num_layers = 12
hidden_size = 64
batch_size = 256
seq_len = 12
features = 5
embedding_dim = 2
isSchedulerMetricBased = False

noteEncoder = joblib.load('config_chords/noteEncoder.pkl')
octaveEncoder = joblib.load('config_chords/octaveEncoder.pkl')

noteNameOneHotEncoder = joblib.load('config_chords/noteNameOneHotEncoder.pkl')
octaveOneHotEncoder = joblib.load('config_chords/octaveOneHotEncoder.pkl')
scaler = joblib.load('config_notes/scaler.pkl')
minMax = joblib.load('config_notes/minMax.pkl')
noteNameEncodedOffset, octaveEncodedOffset = joblib.load('config_notes/offsets.pkl')


def note_name_to_pitch_class(note_name):
    pitch_class_dict = {
        'C': 0, 'C#': 1, 'Db': 1, 'D-': 1,
        'D': 2, 'D#': 3, 'Eb': 3, 'E-': 3,
        'E': 4, 'Fb': 4, 'E#': 5, 'F-': 4,
        'F': 5, 'F#': 6, 'Gb': 6, 'G-': 6,
        'G': 7, 'G#': 8, 'Ab': 8, 'A-': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B-': 10,
        'B': 11, 'Cb': 11, 'B#': 0, 'C-': 11
    }
    return pitch_class_dict[note_name]


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
    # convert to a dataframe
    notes = raw_note_buffer

    noteNames = notes['ChordNotes'].apply(lambda x: [note['NoteName'] for note in x])
    pitches = notes['ChordNotes'].apply(lambda x: [note_name_to_pitch_class(note['NoteName']) for note in x])

    dissonance_scores = []
    instability_scores = []
    if len(pitches) > 0:
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


def _calculate_dissonance(notes):
    """Calculates the dissonance of intervals between consecutive notes."""
    dissonance_map = {1: 0.9, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.1, 6: 0.4, 7: 0.9}
    dissonance_scores = []

    for pitches in notes:
        bits = []
        for i in range(len(pitches) - 1):
            interval = abs(pitches[i + 1] - pitches[i]) % 12  # Interval within an octave
            bits.append(dissonance_map.get(interval, 0))  # Default to 0 if interval not found
        dissonance_scores.append(np.mean(bits))
    return dissonance_scores


def _calculate_instability(notes):
    """Measures tension based on distance from tonal centers (stability)."""
    # Simplified model: assuming C major
    stable_notes = [0, 2, 4, 5, 7, 9, 11]  # Pitch classes of C major scale
    instability_scores = []
    for pitches in notes:
        bits = []
        for pitch in pitches:
            distance = min((pitch - note) % 12 for note in stable_notes)
            # Map distance to instability (example, could be refined)
            instability = (1 / (1 + np.exp(-distance + 3)))
            bits.append(instability)
        instability_scores.append(np.mean(bits))
    return instability_scores


def _calculate_rhythmic_complexity(notes):
    """Scores tension based on variations in note durations."""
    durations = notes['Duration'].astype(float)
    duration_diffs = np.abs(np.diff(durations))
    return np.clip(duration_diffs, a_min=0, a_max=1)  # Normalize values


def _calculate_dynamic_variation(notes):
    """Scores tension based on changes in note velocity (dynamics)."""
    velocities = notes['Velocity'].astype(int)
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
        window_size = self.window_size # np.random.randint(5, self.window_size + 1)

        categorical_in = self.categorical[idx: idx + window_size]
        categorical_out = self.categorical[idx + window_size]
        continuous_in = self.continuous[idx: idx + window_size]
        continuous_out = self.continuous[idx + window_size]

        categorical_in = torch.stack(categorical_in)

        time_steps = categorical_in.shape[0]
        if time_steps > 1:
            # Randomly select a timestep to mask, excluding the last one
            timestep_to_mask = random.randint(0, time_steps - 2)

            categorical_in[timestep_to_mask] = torch.zeros_like(categorical_in[timestep_to_mask])

        return (categorical_in, Tensor(continuous_in)), (Tensor(categorical_out), Tensor(continuous_out))


if rebuild_cache:
    notes_df = pd.read_parquet('allChords.parquet')
    notes_df = notes_df.reset_index(drop=True).reset_index(drop=True).head(1000)

    print(notes_df.columns)

    # drop rows where Velocity is 0
    notes_df = notes_df[notes_df['Velocity'] != 0]

    # set PitchClass to 0 where Velocity is 0
    notes_df.loc[notes_df['Velocity'] == 0, 'PitchClass'] = 0

    # calculate the tension in a future looking window of 12 notes
    # add a new column to the dataframe that contains the avg tension for the next 12 notes
    # tension = []
    # for i in range(len(notes_df)):
    #    if i + 12 < len(notes_df):
    #        tension.append(calculate_tension(notes_df.iloc[i:i+12].values))
    #    else:
    #        tension.append(0)

    # remove rows with empty ChordNotes
    notes_df = notes_df[notes_df['ChordNotes'].apply(lambda x: len(x) <= 6)]

    notes_df['Tension'] = 0.3

    notes_df['Duration'] = notes_df['ChordNotes'].apply(lambda x: max(note['Duration'] for note in x))
    notes_df['Velocity'] = notes_df['ChordNotes'].apply(lambda x: max(note['Velocity'] for note in x))

    tension = calculate_tension(notes_df)

    # order the chordnotes by notename
    notes_df['ChordNotes'] = notes_df['ChordNotes'].apply(lambda x: sorted(x, key=lambda y: y['NoteName']))

    notes_df['Notes2'] = notes_df['ChordNotes'].apply(lambda x: [noteEncoder.transform([n['NoteName']])[0] for n in x])
    notes_df['Octaves2'] = notes_df['ChordNotes'].apply(lambda x: [octaveEncoder.transform([n['Octave']])[0] for n in x])
    notes_df = notes_df.drop(columns=['PitchClass', 'ChordNotes', 'Offset'])

    # padd the notes and octaves to 6
    notes_df['Notes_Padded'] = notes_df['Notes2'].apply(lambda x: x + [0] * (6 - len(x)))
    notes_df['Octaves_Padded'] = notes_df['Octaves2'].apply(lambda x: x + [0] * (6 - len(x)))

    notes_df['Notes_Padded'] = notes_df['Notes_Padded'].apply(lambda x: [noteNameOneHotEncoder.transform([[n]]) for n in x])
    notes_df['Octaves_Padded'] = notes_df['Octaves_Padded'].apply(lambda x: [noteNameOneHotEncoder.transform([[n]]) for n in x])

    notes_df['Notes_Padded'] = notes_df['Notes_Padded'].apply(lambda x: [n[0] for n in x])
    notes_df['Octaves_Padded'] = notes_df['Octaves_Padded'].apply(lambda x: [n[0] for n in x])

    notes_df['Pairs'] = notes_df.apply(lambda x: list(zip(x['Notes_Padded'], x['Octaves_Padded'])), axis=1)

    # list to ndarray

    # drop the offset column

    categorical_vars = notes_df[['Pairs']].values

    def tmpp(x):
        return torch.tensor(x[0])

    octaves_list = [torch.tensor(tmpp(pair), dtype=torch.int16) for pair in categorical_vars]

    categorical_vars = octaves_list
    continuous_vars = notes_df[['Duration', 'Velocity', 'Tension']].values

    # Normalize continuous variables
    continuous_vars = scaler.transform(continuous_vars).astype(np.float16)
    continuous_vars = minMax.transform(continuous_vars).astype(np.float16)

    joblib.dump(categorical_vars, 'data/categorical_chords_vars2.pkl')
    joblib.dump(continuous_vars, 'data/continuous_chords_vars2.pkl')
else:
    # if USE_MINIBATCH env var is 1, use the small dataset
    if os.environ.get('USE_MINIBATCH') == '1':
        categorical_vars = joblib.load('data/categorical_chords_vars_small.pkl')
        continuous_vars = joblib.load('data/continuous_chords_vars_small.pkl')
    else:
        categorical_vars = joblib.load('data/categorical_chords_vars.pkl')
        continuous_vars = joblib.load('data/continuous_chords_vars.pkl')

    #joblib.dump(categorical_vars, 'data/categorical_chords_vars_small.pkl')
    #joblib.dump(continuous_vars, 'data/continuous_chords_vars_small.pkl')

dataset = MusicalNotesDataset(categorical_vars, continuous_vars, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_continuous = 3  # Number of continuous variables
offsets = (octaveEncodedOffset, noteNameEncodedOffset)  # Number of categorical variables
category_sizes = [len(octaveEncoder.classes_), len(noteEncoder.classes_)]

model = MusicPredictionChordModel(num_continuous, offsets, hidden_size, num_layers, embedding_dim, device).to(device)

print(model)

continuousLossFn = nn.HuberLoss()
categoricalLossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10000, factor=0.9, verbose=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001)

eval_loss = 0
continuous_loss = 0
categorical_loss = 0
best_loss = np.inf

# Get sequence lengths
sequence_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

for epoch in range(num_epochs):
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, batch in pbar:
        x = batch[0]
        y = batch[1]
        x_categorical, x_continuous = x

        y_categorical, y_continuous = y

        x_categorical = x_categorical.long().to(device)
        x_continuous = x_continuous.float().to(device)
        y_categorical = y_categorical.float().to(device)
        y_continuous = y_continuous.float().to(device)

        hidden_state = model.init_hidden(batch_size)

        # Forward pass
        categorical_outputs, continuous_output, hidden_state = model(x_categorical, x_continuous, sequence_lengths,
                                                                     hidden_state)

        # Loss calculation
        loss = 0
        continuous_loss = continuousLossFn(continuous_output, y_continuous)

        categorical_loss = 0

        # Loop over each note
        for note_index in range(categorical_outputs.shape[
                                    1]):  # Assuming shape [batch, num_notes, num_features, feature_one_hot_encoded]
            # Compute loss for each note's features individually
            note_categorical_outputs = categorical_outputs[:, note_index, :, :]
            note_y_categorical = y_categorical[:, note_index, :, :]

            octave = note_categorical_outputs[:, 0, :]
            noteName = note_categorical_outputs[:, 1, :]

            y_octave = note_y_categorical[:, 0, :]
            y_noteName = note_y_categorical[:, 1, :]

            categorical_join = [octave, noteName]
            y_join = [y_octave, y_noteName]

            note_loss = sum(categoricalLossFn(xx, yy) for xx, yy in zip(categorical_join, y_join))

            # Compute and accumulate loss for each note
            categorical_loss += note_loss

        # Average the categorical loss across all notes
        categorical_loss /= categorical_outputs.shape[1]

        loss = continuous_loss + categorical_loss
        pbar.set_description(
            f'Epoch {epoch + 1}, Total Loss: {loss.item():.6f}, Cont Loss: {continuous_loss:.6f}, Cat Loss: {categorical_loss:.6f}, Eval Loss: {eval_loss:.6f}, Best Loss: {best_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        loss.backward()

        # Update weights
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        continuous_loss = continuous_loss.item()
        categorical_loss = categorical_loss.item()

        if idx % 1000 == 0 and loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'chord_model/gru_best_model_state.pth')
            torch.save(model.state_dict(),
                       'chord_model/gru_model_epoch_' + str(epoch) + '_idx_' + str(idx) + '_loss_' + str(loss.item()) + '.pth')
            print('Saved checkpoint at loss ' + str(loss.item()))

        pbar.update()

        if isSchedulerMetricBased:
            scheduler.step(loss)
        else:
            scheduler.step()
