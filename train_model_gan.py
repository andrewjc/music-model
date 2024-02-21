import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MusicalNotesDataset(Dataset):
    def __init__(self, data, window_size=5):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx: idx + self.window_size].values
        target = self.data.iloc[idx + self.window_size].values
        # Encoding, normalization, etc.
        return Tensor(sequence), Tensor(target)

class StackedGRU(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size=6, output_size=6, num_categorical=3, num_continuous=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.ff_n = nn.BatchNorm1d(input_size)

        # Stack multiple GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # Output layers for categorical and continuous variables
        self.output_layer_categorical = nn.Linear(hidden_size, num_categorical)
        self.output_layer_continuous = nn.Linear(hidden_size, num_continuous)

        self.init_weights()

    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
        nn.init.xavier_normal_(self.output_layer_categorical.weight.data)
        self.output_layer_categorical.bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.output_layer_continuous.weight.data)
        self.output_layer_continuous.bias.data.fill_(0.01)

    def forward(self, x, hidden=None):
        # Initialize hidden state (if not provided)
        if hidden is None:
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)

        x = self.ff_n(x)

        # Iterate through GRU layers
        x, hidden = self.gru(x, hidden)

        # Final output layers for a sequence length of 1
        out_categorical = self.output_layer_categorical(x[:, -1, :])  # Take the last timestep output
        out_continuous = self.output_layer_continuous(x[:, -1, :])  # Take the last timestep output

        out_categorical = torch.relu(out_categorical)
        out_continuous = torch.relu(out_continuous)

        out_categorical = torch.nn.functional.one_hot(out_categorical, num_classes=self.num_categorical)

        return (out_categorical, out_continuous), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data  # Use weight of a parameter for initialization
        hidden = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden
class DiscriminatorNetwork(nn.Module):
    def __init__(self, hidden_size, input_size=6, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.ff_n = nn.BatchNorm1d(input_size)

        # Stack multiple GRU layers
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
                layer.bias.data.fill_(0.01)


    def forward(self, x):

        x = self.ff_n(x)

        out = self.nn(x)
        return out


# Training
num_epochs = 100000
notes_df = pd.read_parquet('allNotes.parquet')

# encode NoteName to categorical encoding
noteEncoder = LabelEncoder()
notes_df['NoteName'] = noteEncoder.fit_transform(notes_df['NoteName'])
pitchClassEncoder = LabelEncoder()
notes_df['PitchClass'] = pitchClassEncoder.fit_transform(notes_df['PitchClass'])

# drop rows where Velocity is 0
notes_df = notes_df[notes_df['Velocity'] != 0]

# drop the offset column
notes_df = notes_df.drop(columns=['Offset'])

# normalize the data
#notes_df = (notes_df - notes_df.mean()) / notes_df.std()


num_layers = 5
hidden_size = 64
batch_size = 256
seq_len = 12
features = 5

dataset = MusicalNotesDataset(notes_df, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = StackedGRU(num_layers, hidden_size, features, features).to(device)
loss_fn = nn.HuberLoss()
discriminator = DiscriminatorNetwork(hidden_size, features, 1).to(device)

# Optimizers
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))

# Loss function for binary classification
loss_fn2 = nn.BCELoss()

fixed_noise = torch.randn(1, num_layers, features).to(device)
# Training loop
for epoch in range(num_epochs):
    hbar = tqdm(dataloader)
    best_loss_generator = 100000
    best_loss_discriminator = 100000
    for i, batch in enumerate(hbar):
        X, y = batch
        X, y = X.to(device), y.to(device)

        # Discriminator Training:
        discriminator.zero_grad()

        # Real data
        real_data = X.to(torch.float32)  # Assuming batch[0] is your sequence data
        batch_size = real_data.size(0)
        real_label = torch.ones(batch_size).to(device)  # Labels as 'real'

        real_data = real_data[:, -1, :]  # Take the last timestep output

        output = discriminator(real_data)
        loss_d_real = loss_fn2(output, real_label.unsqueeze(1))
        loss_d_real.backward()
        D_x = output.mean().item()

        # Generated data
        noise = torch.randn(batch_size, num_layers, features).to(device)
        g_hidden = generator.init_hidden(batch_size)
        fake_data, _ = generator(noise, g_hidden)
        fake_label = torch.zeros(batch_size).to(device)  # Labels as 'fake'

        output = discriminator(fake_data.detach())  # Detach to avoid backprop through generator
        loss_d_fake = loss_fn2(output, fake_label.unsqueeze(1))
        loss_d_fake.backward()
        D_G_z1 = output.mean().item()

        #  Total discriminator loss and Update
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.step()

        # Generator Training:
        generator.zero_grad()

        # Use frozen discriminator predictions
        output = discriminator(fake_data)
        loss_g = loss_fn2(output, real_label.unsqueeze(1))  # Aim to get the discriminator to label as 'real'
        loss_g.backward()
        D_G_z2 = output.mean().item()
        optimizer_g.step()

        if loss_g.item() < best_loss_generator:
            best_loss_generator = loss_g.item()
            torch.save(generator.state_dict(), f'generator_best.pt')

        if loss_d.item() < best_loss_discriminator:
            best_loss_discriminator = loss_d.item()
            torch.save(discriminator.state_dict(), f'discriminator_best.pt')


        hbar.set_description(f'Epoch [{epoch}/{num_epochs}], Batch {i}, Loss D: {loss_d.item()}, Loss G: {loss_g.item()}'
                                f'Loss D Real: {loss_d_real.item()}, Loss D Fake: {loss_d_fake.item()}')
        hbar.update()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Batch {i}, Loss D: {loss_d.item()}, Loss G: {loss_g.item()}'
                                f'Loss D Real: {loss_d_real.item()}, Loss D Fake: {loss_d_fake.item()}')
            #print(f'D_x: {D_x}, D_G_z1: {D_G_z1}, D_G_z2: {D_G_z2}')

            with torch.no_grad():
                g_hidden = generator.init_hidden(1)
                fake_data, _ = generator(fixed_noise, g_hidden)
                print(fake_data)
                try:
                    print(noteEncoder.inverse_transform(fake_data[0][4].cpu().unsqueeze(0)))
                except:
                    pass

