
from scipy import stats
import numpy as np
from collections import Counter
import pandas as pd

def return_windowed_df(df, columns, label, skip1 = False, overlap = False):
    grouped = df.groupby("section_id", group_keys=True)[columns]
    stride = 512
    all = pd.DataFrame(columns=columns)
    if columns[:-1] == 'mizan':
        
        skip1 = True
    for section_id, group in grouped:
        
        l = 0
        r = 512
        
        temp = pd.DataFrame(columns=columns)
        
        if skip1 and [x[10] for x in group.values[l:r]][0] == 1:
            for i, c in enumerate(columns):
                temp[c] = [x[i] for x in group.values[:]]
            temp['section_id'] = [section_id + str(l)] * len(temp) 
            all = pd.concat([temp, all])
            continue
        
        while(len(group.values[l:r]) >= 512):
            for i, c in enumerate(columns):
                temp[c] = [x[i] for x in group.values[l:r]]
              
            temp['section_id'] = [section_id + str(l)] * stride  
            all = pd.concat([temp, all])
            l += stride
            r += stride
            
    return all

def extract_feature(sequence, pitch_distr, ql_distr=None):
    """
    Parameters:
        sequence: sequence of note with following individual features ['NoteAndRest', 'timestamp_(scs)', 'measure_number', 'quarterLength', 'bpm', 'beatStrength', 'offset', 'tie', 'isChord' ]
        pitch_distr: pitch distribution of sequence
        ql_distr: rhythm distribution of sequence
    -------------------------------------------------------------------

    Extracts derivated global features from sequence and returns a dictionary
    """
    
    def compute_entropy(sequence):
        pitches = [note[0] for note in sequence]
        pitch_counts = np.bincount([p for p in pitches if p >= 0])  # remove rests (-1)
        pitch_prob = pitch_counts / np.sum(pitch_counts)
        
        return stats.entropy(pitch_prob)

    def compute_entropy_duration(sequence):
        durations = [note[1] for note in sequence]
        duration_counts = np.bincount([p for p in durations if p >=0])  # remove rests (-1)
        duration_prob = duration_counts / np.sum(duration_counts)
        
        return stats.entropy(duration_prob)

    pitches = [note[0] for note in sequence]
    durations = [note[1] for note in sequence]
    timestamps = [note[2] for note in sequence]
    offsets = [note[6] for note in sequence]
    beatStrengths = [note[7] for note in sequence]
    length_section = sequence[0][3]
    key_value = Counter([note[5] for note in sequence]).most_common(1)[0][0]
    bpm = np.array([note[4] for note in sequence]).mean()
    duration_ngrams = [tuple(durations[i:i+3]) for i in range(len(durations)-2)]
    

    return {
    "length_section":  length_section,
    "mean_pitch": np.mean(pitches),
    "mean_offset": np.mean(offsets),
    "mean_strengths": np.mean(beatStrengths),
    "mean_duration": np.mean(durations),
    "mean_timestamp": np.mean(timestamps),
    "note_density": len([note[0] for note in sequence if note[0] != -1]) / len(sequence),
    "amount_rest_compared_toNotes": len([x for x in pitches if x == -1]) / len(sequence),
    "bpm": bpm,
    "repeated_rhythms" :  sum(1 for count in Counter(duration_ngrams).values() if count > 1),
    "std_duration": np.std(durations),
    "std_timestamps": np.std([note[2] for note in sequence]),
    "repeats": sum(pitches[i] == pitches[i+1] for i in range(len(pitches)-1)),
    "duration_changes": np.count_nonzero(np.diff(durations)),
    **{f"pitch_pitch_distr_{i}": val for i, val in enumerate(pitch_distr)},
    **{f"ql_pitch_distr_{i}": val for i, val in enumerate(ql_distr)},
    "note_entropy" : compute_entropy(sequence),
    "duration_entropy" : compute_entropy_duration(sequence),
    "key": key_value,
    }

def note_to_midi(note, include_octave=False):
    """
    Parameters:
        note: note in string format to convert. E.g. 'C4', 'B-5', ...
        include_octave: if this flag is true, the octave is take into calculation
        otherwise, we take the octave as 4
    
    -------------------------------------------------------------------

    Convert a note in string format into the corresponding midi pitch
    """
    NOTE_TO_MIDI = {
        'C': 0, 'C#': 1, 'D-': 1, 'C##': 2, 'D--': 1,
        'D': 2, 'D#': 3, 'E-': 3, 'D##': 4, 'E--': 3,
        'E': 4, 'F-': 4, 'E#': 5, 'F': 5, 'F#': 6, 'G-': 6, 'F##': 7, 'G--': 6,
        'G': 7, 'G#': 8, 'A-': 8, 'G##': 9, 'A--': 8,
        'A': 9, 'A#': 10, 'B-': 10, 'A##': 11, 'B--': 10,
        'B': 11, 'C-': 11, 'B#': 0, 'C--': 11
    }

    note_name = note[0:]  
    if include_octave:
        octave = int(note[-1])  # Extract the octave (e.g., '0', '2', etc.)
    else:
        octave = 4  # Extract the octave (e.g., '0', '2', etc.)
    if 'Rest' in note_name: return -1
    return NOTE_TO_MIDI[note_name] + 12 * (octave + 1)