import joblib
import pandas as pd

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

noteEncoder = LabelEncoder()
octaveEncoder = LabelEncoder()

noteNameOneHotEncoder = OneHotEncoder(sparse_output=False)
octaveOneHotEncoder = OneHotEncoder(sparse_output=False)
noteNameOneHotEncoder.sparse = False

scaler = joblib.load('config_notes/scaler.pkl')
minMax = joblib.load('config_notes/minMax.pkl')
noteNameEncodedOffset, octaveEncodedOffset = joblib.load('config_notes/offsets.pkl')

notes_df = pd.read_parquet('allChords.parquet')
notes_df = notes_df.reset_index(drop=True).reset_index(drop=True)

print(notes_df.columns)

# drop rows where Velocity is 0
notes_df = notes_df[notes_df['Velocity'] != 0]

# set PitchClass to 0 where Velocity is 0
notes_df.loc[notes_df['Velocity'] == 0, 'PitchClass'] = 0

# remove rows with empty ChordNotes
notes_df = notes_df[notes_df['ChordNotes'].apply(lambda x: len(x) > 5)]

notes_df['Duration'] = notes_df['ChordNotes'].apply(lambda x: max(note['Duration'] for note in x))
notes_df['Velocity'] = notes_df['ChordNotes'].apply(lambda x: max(note['Velocity'] for note in x))

# order the chordnotes by notename
notes_df['ChordNotes'] = notes_df['ChordNotes'].apply(lambda x: sorted(x, key=lambda y: y['NoteName']))

allNotes = []
allOctaves = []

def process(x):
    for n in x:
        if n['NoteName'] not in allNotes:
            allNotes.append(n['NoteName'])
        if n['Octave'] not in allOctaves:
            allOctaves.append(n['Octave'])
    return x

notes_df['ChordNotes'].apply(lambda x: process(x))

# convert list of notes to a dataframe
allNotes = pd.DataFrame(allNotes, columns=['NoteName'])
allOctaves = pd.DataFrame(allOctaves, columns=['Octave'])

allNotes = noteEncoder.fit_transform(allNotes['NoteName'].values.reshape(-1, 1))
allNotes = noteNameOneHotEncoder.fit_transform(allNotes.reshape(-1, 1))

allOctaves = octaveEncoder.fit_transform(allOctaves['Octave'].values.reshape(-1, 1))
allOctaves = octaveOneHotEncoder.fit_transform(allOctaves.reshape(-1, 1))

joblib.dump(noteEncoder, 'config_chords/noteEncoder.pkl')
joblib.dump(octaveEncoder, 'config_chords/octaveEncoder.pkl')
joblib.dump(noteNameOneHotEncoder, 'config_chords/noteNameOneHotEncoder.pkl')
joblib.dump(octaveOneHotEncoder, 'config_chords/octaveOneHotEncoder.pkl')

print("Encoders build.")
exit(0)
