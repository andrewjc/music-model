import joblib
import pandas as pd
import numpy as np
import os

from adaml import AdamL

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_scaling = hidden_size ** -0.5

    def forward(self, gru_out):
        Q = self.query_proj(gru_out)
        K = self.key_proj(gru_out)
        V = self.value_proj(gru_out)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attention_scaling
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        return output

class MusicPredictionChordModel(nn.Module):
    def __init__(self, num_continuous, category_offsets, hidden_size, num_layers, embedding_dim, device):
        super(MusicPredictionChordModel, self).__init__()
        self.num_continuous = num_continuous
        self.category_offsets = category_offsets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.num_categories = sum(category_offsets)

        # Embedding layers for categorical variables
        embedding_output_size = embedding_dim
        self.embeddings = nn.Embedding(self.num_categories, embedding_output_size).to(device)

        # Linear layer for continuous variables
        self.continuous_linear = nn.Linear(num_continuous, hidden_size)
        self.inst_norm_1 = nn.LayerNorm(hidden_size, elementwise_affine=True)

        self.combined_linear = nn.Linear(352, 64)
        self.inst_norm_2 = nn.LayerNorm(64, elementwise_affine=True)

        # GRU layer
        self.gru = nn.GRU(64, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2, bidirectional=True)

        # Output layer for n+1 predictions
        self.head_continuous = nn.Linear(192, num_continuous, bias=False)
        self.head_categorical = nn.Linear(192, 6 * 2 * 12, bias=False)

        self.inst_norm_3 = nn.LayerNorm(192, elementwise_affine=True)

        self.categorical_ln = nn.LayerNorm(448, elementwise_affine=True)
        self.continuous_ln = nn.LayerNorm(448, elementwise_affine=True)

        self.attention_layer = AttentionLayer(192)

        self.init_weights()



    def forward(self, x_categorical, x_continuous, sequence_lengths, hidden_state):
        batch_size, seq_len, num_notes_in_chord, num_features, num_bits = x_categorical.size()

        embedded_outputs = []

        for t in range(seq_len):
            for n in range(num_notes_in_chord):
                current_slice = x_categorical[:, t, n, :, :]
                current_slice_flat = current_slice.view(batch_size, -1)
                embedded_slice = self.embeddings(current_slice_flat)
                embedded_outputs.append(embedded_slice)

        embedded_outputs = torch.stack(embedded_outputs, dim=1)

        x_categorical_embedded = embedded_outputs.view(batch_size, seq_len, -1)

        x_continuous = self.inst_norm_1(F.relu(self.continuous_linear(x_continuous)))

        x_combined = torch.cat([x_continuous, x_categorical_embedded], dim=2)
        x_combined = self.inst_norm_2(F.relu(self.combined_linear(x_combined)))

        x_packed = nn.utils.rnn.pack_padded_sequence(x_combined, sequence_lengths, batch_first=True, enforce_sorted=False)
        gru_out, hidden_state = self.gru(x_packed, hidden_state)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        gru_comb = torch.cat([gru_out, x_combined], dim=2)

        gru_out = self.inst_norm_3(F.relu(gru_comb))

        x_combined = self.attention_layer(gru_out)

        cat_output = self.head_categorical(x_combined)[:, -1, :]
        cont_output = self.head_continuous(x_combined)[:, -1, :]

        # Apply softmax to categorical outputs for probabilities
        #octave_probabilities = F.softmax(octave, dim=-1)
        #note_name_probabilities = F.softmax(noteName, dim=-1)

        cat_output = cat_output.view(-1, 6, 2, 12)

        return cat_output, cont_output, hidden_state

    def init_weights(self):
        # Embedding layers for categorical variables
        nn.init.xavier_normal_(self.embeddings.weight.data)

        # Linear layer for continuous variables
        nn.init.xavier_normal_(self.continuous_linear.weight.data)
        nn.init.zeros_(self.continuous_linear.bias.data)

        # Linear layer for combined linear layer
        nn.init.xavier_normal_(self.combined_linear.weight.data)
        nn.init.zeros_(self.combined_linear.bias.data)


        # GRU layer
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            else:
                nn.init.zeros_(param.data)


        # Output heads for continuous and categorical:
        nn.init.xavier_normal_(self.head_continuous.weight.data)
        nn.init.xavier_normal_(self.head_categorical.weight.data)

        # Initialize batch norm layers
        nn.init.zeros_(self.inst_norm_1.bias.data)
        nn.init.ones_(self.inst_norm_1.weight.data)
        nn.init.zeros_(self.inst_norm_2.bias.data)
        nn.init.ones_(self.inst_norm_2.weight.data)
        nn.init.zeros_(self.inst_norm_3.bias.data)
        nn.init.ones_(self.inst_norm_3.weight.data)

        nn.init.zeros_(self.categorical_ln.bias.data)
        nn.init.ones_(self.categorical_ln.weight.data)
        nn.init.zeros_(self.continuous_ln.bias.data)
        nn.init.ones_(self.continuous_ln.weight.data)

        # Initialize attention layer
        nn.init.xavier_normal_(self.attention_layer.query_proj.weight.data)
        nn.init.xavier_normal_(self.attention_layer.key_proj.weight.data)
        nn.init.xavier_normal_(self.attention_layer.value_proj.weight.data)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data  # Use weight of a parameter for initialization
        hidden = weight.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
        return hidden
