from scipy import stats
from collections import Counter
import numpy as np
import pandas as pd
from utilities.constants import *
from utilities.model_matching_computation import *
from utilities.temporal_analysis import get_folded_rhythm_histogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

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
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) -1)]
    num_ascending_intervals = len([interval for interval in intervals if interval > 0])
    dominant_interval = Counter(intervals).most_common(1)[0][0]
    key_value = Counter([note[5] for note in sequence]).most_common(1)[0][0]
    bpm = np.array([note[4] for note in sequence]).mean()
    duration_ngrams = [tuple(durations[i:i+3]) for i in range(len(durations)-2)]
    sum_pitch_changes = sum([1 if (pitches[i] < pitches[i+1] and pitches[i+1] > pitches[i+2]) or
                        (pitches[i] > pitches[i+1] and pitches[i+1] < pitches[i+2])  else 0 for i in range(len(pitches) -2)])

    sum_duration_changes = sum([1 if (pitches[i] < pitches[i+1] and pitches[i+1] > pitches[i+2]) or
                        (pitches[i] > pitches[i+1] and pitches[i+1] < pitches[i+2])  else 0 for i in range(len(pitches) -2)])
    number_distinct_melodic_intervals = len(np.unique(np.array(intervals)))
    counter = Counter(intervals)


    fixed_intervals = list(range(-11, 12))  # from -11 to 11 inclusive

    # Step 3: Build the fixed-size array
    histogram_array = [counter.get(i, 0) for i in fixed_intervals]

    return {
    "length_section":  length_section,
    "mean_pitch": np.mean(pitches),
    #"pitch_range": np.max(pitches) - np.min(pitches),
    "pitch_changes": sum_pitch_changes,
    "mean_offset": np.mean(offsets),
    "mean_strengths": np.mean(beatStrengths),
    "mean_duration": np.mean(durations),
    "duration_changes": sum_duration_changes,
    #"duration_range": np.max(durations) - np.min(durations),
    #"dominant_melodic_interval" : dominant_interval,
    "num_ascending_intervals" : num_ascending_intervals,
    "number_distinct_melodic_intervals" : number_distinct_melodic_intervals,
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
    **{f"interval_distr{i}": val for i, val in enumerate(histogram_array)},
    "note_entropy" : compute_entropy(sequence),
    "duration_entropy" : compute_entropy_duration(sequence),
    "key": key_value,
    }

"""
def normalize_and_traspose(df_toNorm, label, pitch_distr_sections, ql_distr_sections):
    le = LabelEncoder()
    # Extract global features for each sequence
    grouped = df_toNorm.groupby("section_id", group_keys=True)[['NoteAndRest', 'quarterLength', 'timestamp_(scs)', 'length_section', 'bpm', 'key', 'offset', 'beatStrength']]
    group_arrays = {section_id: group.values for section_id, group in grouped}
    avg_sequence_length = 0
    sequences_features = []
    for key, values in group_arrays.items():
        sequences_features.append(extract_feature(values, pitch_distr_sections[key], ql_distr_sections[key]))
        avg_sequence_length += len(values)

    sequences_features = [list(x.values()) for x in sequences_features]
    # Normalize global features
    scaler = StandardScaler()
    sequences_features = np.array(sequences_features)
    index = list(range((sequences_features.shape[1]) -1))
    features_toNormalize = sequences_features[:, index]
    normalized_features = scaler.fit_transform(features_toNormalize)
    # Replace the original features with the normalized ones
    sequences_features[:, index] = normalized_features
    # Unique to dl
    mask = df_toNorm[df_toNorm['NoteAndRest'] != -1]
    df_toNorm.loc[mask.index, 'NoteAndRest'] = (
        df_toNorm.loc[mask.index, 'NoteAndRest'] - df_toNorm.loc[mask.index, 'NoteAndRest'].mean()
    ) / df_toNorm.loc[mask.index, 'NoteAndRest'].std()

    keys_to_encode = df_toNorm['key'].apply(lambda x: str(x)).tolist()
    df_toNorm['key'] =  le.fit_transform(keys_to_encode)
    df_toNorm['tie'] = df_toNorm['tie'].apply(lambda x: 1 if type(x) ==str else 0)
    df_toNorm['is_rest'] = [1 if x == -1 else 0 for x in df_toNorm['NoteAndRest'].tolist()]

    # NORMALIZATION
    def normalize_columns(df_toNorm, to_norm):
        for column_to_normalize in to_norm:
            df_toNorm[column_to_normalize] = (df_toNorm[column_to_normalize] - df_toNorm[column_to_normalize].mean()) / df_toNorm[column_to_normalize].std()
    normalize_columns(df_toNorm, ['bpm', 'length_section', 'beatStrength', 'offset', 'quarterLength'])

    def transpose_df_toNorm(df_toNorm, columns, label):
        grouped = df_toNorm.groupby("section_id", group_keys=True)[columns]
        all = []
        for section_id, group in grouped:
            all.append([[[x[0], x[1], x[3], x[4], x[5], 1 if x[0] == -1 else 0]for x in group.values[:]] , section_id, group.values[0][2]] )

        return pd.DataFrame(all, columns=['NoteAndRest', 'section_id', label])

    df_toNorm = transpose_df_toNorm(df_toNorm,  ['NoteAndRest', 'quarterLength', label, 'offset', 'beatStrength', 'is_rest'], label)
    df_toNorm['global_features'] = list(sequences_features)
    # Reset labels into range (0, n)
    df_toNorm[label] = df_toNorm[label].apply(lambda x: replace_label(x, label))
    return df_toNorm
"""

def normalize_and_traspose(df_toNorm, label, pitch_distr_sections, ql_distr_sections):
    le = LabelEncoder()
    # Extract global features for each sequence
    grouped = df_toNorm.groupby("section_id", group_keys=True)[['NoteAndRest', 'quarterLength', 'timestamp_(scs)', 'length_section', 'bpm', 'key', 'offset', 'beatStrength']]
    group_arrays = {section_id: group.values for section_id, group in grouped}
    avg_sequence_length = 0
    sequences_features = []
    for key, values in group_arrays.items():
        sequences_features.append(extract_feature(values, pitch_distr_sections[key], ql_distr_sections[key]))
        avg_sequence_length += len(values)

    sequences_features = [list(x.values()) for x in sequences_features]

    sequences_features = np.array(sequences_features)
    df_toNorm['is_rest'] = [1 if x == -1 else 0 for x in df_toNorm['NoteAndRest'].tolist()]
    keys_to_encode = df_toNorm['key'].apply(lambda x: str(x)).tolist()
    df_toNorm['key'] =  le.fit_transform(keys_to_encode)
    df_toNorm['tie'] = df_toNorm['tie'].apply(lambda x: 1 if type(x) ==str else 0)

    def transpose_df_toNorm(df_toNorm, columns, label):
        grouped = df_toNorm.groupby("section_id", group_keys=True)[columns]
        all = []
        for section_id, group in grouped:
            all.append([[[x[0], x[1], x[3], x[4], x[5], 1 if x[0] == -1 else 0]for x in group.values[:]] , section_id, group.values[0][2]] )

        return pd.DataFrame(all, columns=['NoteAndRest', 'section_id', label])

    df_toNorm = transpose_df_toNorm(df_toNorm,  ['NoteAndRest', 'quarterLength', label, 'offset', 'beatStrength', 'is_rest'], label)
    df_toNorm['global_features'] = list(sequences_features)
    # Reset labels into range (0, n)
    df_toNorm[label] = df_toNorm[label].apply(lambda x: replace_label(x, label))
    return df_toNorm

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

def midi_to_note(midi_val, include_octave=False):
    """
    Convert a MIDI pitch number into a note string.

    Parameters:
        midi_val (int): MIDI pitch number (0–127), or -1 for Rest.
        include_octave (bool): Whether to include octave number in the result.

    Returns:
        str: Note string (e.g., 'C', 'G#4', 'Rest')
    """
    if midi_val == -1:
        return 'Rest'

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    note_num = midi_val % 12
    octave = (midi_val // 12) - 1  # MIDI standard: C4 = 60

    note = NOTE_NAMES[note_num]
    return f"{note}{octave}" if include_octave else note

def compare_plot_label_distribution(df1, df2, plot = True):
    """
    Parameters:
        df1: First Dataframe to compare
        df2: Second Dataframe to compare
        plot: If True, both distributions are plotted in a bar chart

    -------------------------------------------------------------------

    Returns the distributions of the labels ['nawba', 'tab', 'mizan'] in the different dataframes of the same structure.
    If the Plot flag is true, it also displays in a bar chart the distributions
    """
    def return_label_distributions(df):
        distributions = {
            'nawba' : [],
            'tab' : [],
            'mizan' : []
        }
        for l2 in ['nawba', 'tab', 'mizan']:

            for l in LABEL_LIST_TRAIN[l2]:

                distributions[l2].append(len(df[df[l2] == l]['section_id'].unique()))
        return distributions
    if plot:
        fig = plt.figure(figsize=(20, 4))
        for i, l2 in enumerate(['nawba', 'tab', 'mizan']):
            plt.subplot(1, 3, i+1)
            plt.bar(LABEL_LIST_TRAIN[l2], (return_label_distributions(df1)[l2]), alpha = 0.8, label = 'New Sections')
            plt.bar(LABEL_LIST_TRAIN[l2], (return_label_distributions(df2)[l2]), alpha = 0.8, label = 'Original')
        plt.tight_layout()
        plt.legend()
    return return_label_distributions(df1), return_label_distributions(df2)

def replace_label(x, label):
    """
    Parameters:
        x: Dataframe for which we substitute the labels
        label: Specific label for which we substitute the values. Can be either one of ['nawba', 'tab', 'mizan']

    -------------------------------------------------------------------

    Replaces in the label column of the dataframe x the values so that each classo is represented in an ordinal value from (0, n).
    For example: the labels [1, 2, 6, 8] will be substitued to [0, 1, 2, 3]
    """
    return REPLACE_STRING[label][str(int(x))]


def return_windowed_df(df, columns, label, skip1 = False, stride_percentage = 0.5, length_sequence = 1024):
    """
    Parameters:
        df: Dataframe to section
        label: ....
        skip1: ....
        apply_stride: If True, consecutive sections will have an overlapping stride
        of half the sequence length
        length_sequence: Note Length of each sequence

    -------------------------------------------------------------------

    Returns the Dataframe with splitted section based on the length_sequence parameter:
    A section id with x notes will be split in x/length_sequence sections
    """
    grouped = df.groupby("section_id", group_keys=True)[columns]
    stride_percentage = 1 - stride_percentage
    stride = length_sequence * stride_percentage if stride_percentage > 0  else length_sequence
    float_stride = stride
    stride = int(stride)
    all = pd.DataFrame(columns=columns)
    if columns[:-1] == 'mizan':

        skip1 = True
    for section_id, group in grouped:

        l = 0
        r = length_sequence
        temp = pd.DataFrame(columns=columns)

        if skip1 and [x[10] for x in group.values[l:r]][0] == 1:
            for i, c in enumerate(columns):
                temp[c] = [x[i] for x in group.values[:]]
            temp['section_id'] = [section_id + str(l)] * len(temp)
            all = pd.concat([temp, all])
            continue

        while(len(group.values[l:r]) >= length_sequence):
            for i, c in enumerate(columns):
                temp[c] = [x[i] for x in group.values[l:r]]


            temp['section_id'] = [section_id + str(l)] *  len(temp['NoteAndRest'])#int(float_stride * (length_sequence / stride))
            all = pd.concat([temp, all])
            l += stride
            r += stride
    return all

def prepare_dataframe(label, overlap, sequence_length):
    le = LabelEncoder()
    df_notes = pd.read_json('note_corpus3.json', orient ='split', compression = 'infer')
    #df_notes = get_timestamps_from_all_corpus()
    df_notes['tie'] = df_notes['tie'].fillna(-1)
    df_notes = df_notes.dropna()

    df_notes = df_notes[~df_notes["section_id"].str.contains('0386e377-7212-43e5-89b6-7f4c42d0ae74', na=False)]
    df_notes['NoteAndRest'] = df_notes['NoteAndRest'].apply(lambda x: re.sub(r'\d+', '', x))
    df_notes['quarterLength'] = [0.33 if i == '1/3' else i for i in df_notes['quarterLength']]
    df_notes['offset'] = [8/3 if i == '8/3' else i for i in df_notes['offset']]
    df_notes['offset'] = [7/3 if i == '7/3' else i for i in df_notes['offset']]
    df_notes['LetterNotes'] = df_notes['NoteAndRest']
    df_notes['NoteAndRest'] = df_notes['NoteAndRest'].apply(lambda x: note_to_midi(x))
    df_notes = label_length(df_notes)

    #df_notes = df_notes[df_notes['length_section'] > 30]
    # Function to convert note and octave to MIDI number
    df_windowed = return_windowed_df(df_notes, ['NoteAndRest', 'LetterNotes', 'quarterLength', 'timestamp_(scs)', 'length_section', 'bpm', 'key', 'offset', 'beatStrength', 'tie', 'nawba', 'mizan', 'tab'], label, stride_percentage= overlap, length_sequence= sequence_length)
    df_windowed = label_length(df_windowed)
    # Create new dataset with translated notes to pitch

    df_midi = df_windowed.copy()

    #df_midi['NoteAndRest'] = df_midi['NoteAndRest'].apply(lambda x: note_to_midi(x))
    df = df_midi[df_midi[label].isin(LABEL_LIST_TRAIN[label])]
    keys_to_encode = df['key'].apply(lambda x: str(x)).tolist()
    df['key'] =  le.fit_transform(keys_to_encode)

    pitch_distr_sections = df_windowed.groupby('section_id').apply(lambda x: list(compute_avg_folded_hist_labeled_notes(x['LetterNotes'].tolist(), x['quarterLength'].tolist())))
    ql_distr_sections = df_windowed.groupby('section_id').apply(lambda x: list(get_folded_rhythm_histogram(x['quarterLength'].tolist())))

    return df, pitch_distr_sections, ql_distr_sections

def return_prefixes(df, label):
    to_stratify = []
    prefix_to_ids = defaultdict(list)
    prev_pref = ""
    for id_ in df['section_id'].unique():
        prefix = id_[:PREFIX_LENGTH]
        prefix_to_ids[prefix].append(id_)
        if prev_pref != prefix:
            to_stratify.append(df.loc[df['section_id'].str.contains(id_)][label].tolist()[0])
            prev_pref = prefix
    prefixes = list(prefix_to_ids.keys())
    prefixes = np.array(prefixes)
    return prefixes, to_stratify, prefix_to_ids


