from fractions import Fraction as fractions
from multiprocessing.pool import Pool

import pandas as pd
from pandas import DataFrame

midi_path = "F:\\Datasets\\midi_files"

import os
import numpy as np
from music21 import converter, instrument, note, chord, clef


# iterate through all midi files and extract notes
def process_file(file):
    try:
        file_path = os.path.join(midi_path, file)
        if file.endswith(".mid"):
            notes, chords = read_midi(file_path)
            notes_df = pd.DataFrame(notes, columns=['PitchClass', 'Octave', 'NoteName', 'Duration', 'Velocity', 'Offset'])
            chords_df = pd.DataFrame(chords, columns=['ChordNotes', 'Duration', 'Velocity', 'Offset'])
            print("Processed file: ", file, " with ", str(len(notes_df)), " notes.")
            return notes_df, chords_df
    except Exception as e:
        print("Error processing file: ", file)
        print(e)
        # delete the file
        os.remove(file_path)
        return DataFrame(), DataFrame()

if __name__ == "__main__":
    max_rows = 10e8
    max_files = 3000

    with Pool() as pool:  # Create a process pool
        results = pool.map(process_file, os.listdir(midi_path)[0:max_files])

        print("Processed all files.. beginning merge")
        all_notes = pd.concat([result[0] for result in results])
        all_chords = pd.concat([result[1] for result in results])

        print("Processed all files with", str(len(all_notes)), "notes")

        if len(all_notes) > max_rows:
            all_notes = all_notes.head(int(max_rows))  # Truncate if necessary

        all_notes.to_parquet('allNotes.parquet')
        all_chords.to_parquet('allChords.parquet')

# read each midi file and extract the notes
def read_midi(file):
    notes = [] #DataFrame(columns=['PitchClass', 'Octave', 'NoteName', 'Duration', 'Velocity', 'Offset'])
    chords = [] #DataFrame(columns=['ChordNotes', 'Duration', 'Velocity', 'Offset'])
    notes_to_parse = None
    # parsing a midi file
    try:
        midi = converter.parse(file)
    except:
        print("Error parsing file: " + file)
        return notes, chords


    try:

        # group based on different instruments
        s2 = instrument.partitionByInstrument(midi)
        # loop over all the instruments
        for part in s2.parts:
            # select elements of only piano
            if 'Drums' not in str(part):
                notes_to_parse = part.recurse()
                # finding whether a particular element is note or a chord
                for element in notes_to_parse:
                    # note
                    if isinstance(element, note.Note):
                        pitchClass = element.pitch.pitchClass
                        octave = element.pitch.octave
                        noteName = element.pitch.name
                        element.duration.consolidate()
                        duration = element.duration.quarterLength
                        if isinstance(duration, fractions):
                            duration = float(duration)
                        if isinstance(duration, float) == False:
                            duration = float(duration)
                        velocity = element.volume.velocity
                        offset = element.offset
                        if isinstance(offset, fractions):
                            offset = float(offset)
                        if isinstance(offset, float) == False:
                            offset = float(offset)

                        rw = {'PitchClass': pitchClass, 'Octave': octave, 'NoteName': noteName, 'Duration': duration, 'Velocity': velocity, 'Offset': offset}
                        notes.append(rw)
                    # chord
                    elif isinstance(element, chord.Chord):
                        element.duration.consolidate()
                        duration = element.duration.quarterLength
                        if isinstance(duration, fractions):
                            duration = float(duration)
                        if isinstance(duration, float) == False:
                            duration = float(duration)
                        offset = element.offset
                        if isinstance(offset, fractions):
                            offset = float(offset)
                        if isinstance(offset, float) == False:
                            offset = float(offset)
                        velocity = element.volume.velocity

                        chordNotes = []
                        for pitch in element.pitches:
                            pitchClass = pitch.pitchClass
                            octave = pitch.octave
                            noteName = pitch.name

                            rw = {'PitchClass': pitchClass, 'Octave': octave, 'NoteName': noteName, 'Duration': duration,
                                  'Velocity': velocity, 'Offset': offset}
                            chordNotes.append(rw)

                        rw = {'ChordNotes': chordNotes, 'Duration': duration, 'Velocity': velocity, 'Offset': offset}
                        chords.append(rw)
                    elif isinstance(element, note.Rest):
                        element.duration.consolidate()
                        duration = element.duration.quarterLength
                        if isinstance(duration, fractions):
                            duration = float(duration)
                        if isinstance(duration, float) == False:
                            duration = float(duration)
                        offset = element.offset
                        if isinstance(offset, fractions):
                            offset = float(offset)
                        if isinstance(offset, float) == False:
                            offset = float(offset)
                        rw = {'PitchClass': 9999, 'Octave': 9999, 'NoteName': 'Rest', 'Duration': duration, 'Velocity': 0, 'Offset': offset}
                        notes.append(rw)
    except:
        print("Error parsing file: " + file)

    return notes, chords
