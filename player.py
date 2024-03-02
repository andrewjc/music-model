import joblib
import pandas as pd
import numpy as np

import curses

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn.functional as F
import sounddevice as sd

from music_model import MusicPredictionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_layers = 3
hidden_size = 64
batch_size = 512
seq_len = 24
features = 5
embedding_dim = 4

noteEncoder = joblib.load('config_notes/noteEncoder.pkl')
octaveEncoder = joblib.load('config_notes/octaveEncoder.pkl')

noteNameOneHotEncoder = joblib.load('config_notes/noteNameOneHotEncoder.pkl')
octaveOneHotEncoder = joblib.load('config_notes/octaveOneHotEncoder.pkl')
scaler = joblib.load('config_notes/scaler.pkl')
minMax = joblib.load('config_notes/minMax.pkl')
noteNameEncodedOffset, octaveEncodedOffset = joblib.load('config_notes/offsets.pkl')

num_continuous = 3  # Number of continuous variables
num_categories = 34  # Number of categorical variables
offsets = (octaveEncodedOffset, noteNameEncodedOffset)  # Number of categorical variables
category_sizes = [len(octaveEncoder.classes_), len(noteEncoder.classes_)]



def prepare(dataframe, tension=0.0):
    dataframe['Octave'] = octaveEncoder.transform(dataframe['Octave'])
    dataframe['NoteName'] = noteEncoder.transform(dataframe['NoteName'])
    # drop rows where Velocity is 0
#    dataframe = dataframe[dataframe['Velocity'] != 0]
    # drop the offset column
    noteNameEncoded = noteNameOneHotEncoder.transform(dataframe['NoteName'].values.reshape(-1, 1))
 #   pitchClassEncoded = pitchClassOneHotEncoder.transform(dataframe['PitchClass'].values.reshape(-1, 1)).toarray()
    octaveEncoded = octaveOneHotEncoder.transform(dataframe['Octave'].values.reshape(-1, 1))
    categorical_vars = np.concatenate((octaveEncoded, noteNameEncoded), axis=1)
    # Normalize continuous variables

    # add the tension column
    dataframe['Tension'] = tension

    continuous_vars = dataframe[['Duration', 'Velocity', 'Tension']].values
    # Normalize continuous variables
    continuous_vars = scaler.transform(continuous_vars)
    continuous_vars = minMax.transform(continuous_vars)

    offsets = (octaveEncoded.shape[1], noteNameEncoded.shape[1])

    return categorical_vars, continuous_vars, offsets


def decode(cont_output, cat_output):
    cont_output = minMax.inverse_transform(cont_output)
    cont_output = scaler.inverse_transform(cont_output)
    # split the categorical_outputs
    octave = cat_output[:, 0:octaveEncodedOffset]
    noteName = cat_output[:, octaveEncodedOffset:]

    octave = F.softmax(torch.Tensor(octave), dim=1)
    noteName = F.softmax(torch.Tensor(noteName), dim=1)

    octave_percentages = octave.numpy() * 100
    noteName_percentages = noteName.numpy() * 100

    octave_predictions = list(zip([octaveEncoder.inverse_transform([o]) for o in range(octave.shape[1])], [float(octave_percentages[:, o]) for o in range(octave_percentages.shape[1])]))
    noteName_predictions = list(zip([noteEncoder.inverse_transform([o]) for o in range(noteName.shape[1])], [float(noteName_percentages[:, o]) for o in range(noteName_percentages.shape[1])]))

    octave_predictions.sort(key=lambda x: x[1], reverse=True)
    noteName_predictions.sort(key=lambda x: x[1], reverse=True)

    cont_output = cont_output[0]
    duration, velocity, tension = cont_output[0], cont_output[1], cont_output[2]

    return octave_predictions, noteName_predictions, duration, velocity, tension


def invokeModel(model, categorical_vars, continuous_vars):
    model.eval()
    hidden = model.init_hidden(1)
    categorical = np.expand_dims(categorical_vars, axis=0)
    continuous = np.expand_dims(continuous_vars, axis=0)
    sequence_lengths = [categorical.shape[1]]
    categorical = torch.Tensor(categorical).to(device).to(torch.int64)
    continuous = torch.Tensor(continuous).to(device).to(torch.float32)
    sequence_lengths = torch.Tensor(sequence_lengths).to(torch.int64)
    hidden = hidden.to(device)
    cont_output, cat_output, hidden = model(categorical, continuous, sequence_lengths, hidden)

    return cont_output, cat_output, hidden


category_offsets = [len(octaveEncoder.classes_), len(noteEncoder.classes_)]
model = MusicPredictionModel(num_continuous, category_offsets, hidden_size, num_layers, embedding_dim, device)
model.to(device)
model.load_state_dict(torch.load('note_model/gru_model_epoch_65_idx_2000_loss_2.5926434993743896.pth'))

tension = 0.3
rawNotebuffer = [[4, 'A', 1.00, 78],
                 [4, 'F', 0.50, 78],
                 [4, 'D', 1.00, 78]]

def draw_note_selector_textonly(stdscr, selected_idx, octave_options, noteName_options, duration, velocity, tension):
    for idx, grp in enumerate(noteName_options):
        noteName, perc = grp
        noteName = noteName[0]
        row_str = f"{str(noteName)} ({perc:.4f})"
        if idx == selected_idx:
            print(f"> {row_str}")  # Indicate the selected option with '>'
        else:
            print(f"  {row_str}")
    print("\n" * (20 - len(noteName_options)))

def draw_note_selector_curses(stdscr, selected_idx, octave_options, noteName_options, duration, velocity, tension):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    for idx, grp in enumerate(noteName_options):
        noteName, perc = grp
        noteName = noteName[0]
        row_str = f"{str(noteName)} ({perc:.4f})"
        x = w//2 - len(row_str)//2
        y = h//2 - len(noteName_options)//2 + idx
        if y < h and x < w:
            if idx == selected_idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row_str)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row_str)
    stdscr.refresh()

def draw_timeline(stdscr, rawNotebuffer, scroll_offset):
    x = 0
    for note in rawNotebuffer:
        octave, noteName, duration, velocity = note
        note_str = f"{noteName}{octave} ({duration:.2f})"
        if x - scroll_offset < stdscr.getmaxyx()[1]:
            stdscr.addstr(0, x - scroll_offset, note_str)
        x += len(note_str)

scroll_offset = 0

draw_note_selector = draw_note_selector_curses

def standardizeNotation(note):
    # if note has a dash, treat it as a flat note
    if '-' in note:
        note = note.replace('-', 'b')
    if len(note) == 2:
        return note[0].upper() + note[1].lower()
    else:
        return note[0]


class NotePlayer:
    def __init__(self):
        self.noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.octave = 4
        self.noteFreqs = self.getNoteFreqs()
        self.sampleRate = 44100
        self.duration = 0.5
        self.velocity = 0.5

    def getNoteFreqs(self):
        noteFreqs = {}
        a = 440
        for i in range(-57, 50):
            noteFreqs[i] = a * (2 ** (i / 12))
        return noteFreqs

    def playNote(self, octave, note, duration, velocity):
        if duration < 0.1:
            duration = 0.5

        flat_notes = {'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#'}
        note = standardizeNotation(note)
        if note in flat_notes:
            note = flat_notes[note]

        noteIdx = self.noteNames.index(note)
        noteFreq = self.noteFreqs[(octave - 4) * 12 + noteIdx]
        t = np.linspace(0, duration, int(self.sampleRate * duration), False)
        note = np.sin(2 * np.pi * noteFreq * t) * velocity
        audio = note * (2 ** 15 - 1) / np.max(np.abs(note))
        audio = audio.astype(np.int16)

        sd.play(audio, self.sampleRate)
        sd.wait()



def main(stdscr):
    curses.curs_set(0)

    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    current_row = 0
    tension = 3.1

    notePlayer = NotePlayer()

    exit_loop = False
    manualSelection = True
    temperature = 0.8
    scroll_offset = 0
    while not exit_loop:
        notes_df = pd.DataFrame(rawNotebuffer, columns=['Octave', 'NoteName', 'Duration', 'Velocity'])

        continuous_vars, categorical_vars, offsets = prepare(notes_df, tension)

        cat_output, cont_output, hidden = invokeModel(model, continuous_vars, categorical_vars)

        octave_options, noteName_options, duration, velocity, tension = decode(cont_output.cpu().detach().numpy(),
                                                                               cat_output.cpu().detach().numpy())

        if manualSelection:
            # Print the menu
            draw_note_selector(stdscr, current_row, octave_options, noteName_options, duration, velocity, tension)
            newNote = None
            newOctave = None
            while newNote is None:
                key = stdscr.getch()

                if key == curses.KEY_UP and current_row > 0:
                    current_row -= 1
                elif key == curses.KEY_DOWN and current_row < len(noteName_options) - 1:
                    current_row += 1
                elif key == curses.KEY_LEFT:
                    scroll_offset = max(0, scroll_offset - 1)
                elif key == curses.KEY_RIGHT:
                    scroll_offset += 1
                elif key == curses.KEY_ENTER or key in [10, 13]:
                    newNote = noteName_options[current_row][0]
                    newOctave = octave_options[0][0]
                    break
                draw_note_selector(stdscr, current_row, octave_options, noteName_options, duration, velocity, tension)
        else:
            # select a random note based on the softmax probabilities and temperature
            noteName_options = noteName_options[:5]
            noteName, perc = noteName_options[0]
            newNote = noteName
            newOctave = octave_options[0][0]

        print(newOctave, newNote, duration, velocity)

        notePlayer.playNote(newOctave[0], newNote[0], duration / 10, velocity)

        rawNotebuffer.append([newOctave[0], newNote[0], duration, velocity])

        draw_timeline(stdscr, rawNotebuffer, scroll_offset)


curses.wrapper(main)