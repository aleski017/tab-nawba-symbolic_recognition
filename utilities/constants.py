LIST_NOTES = ['C', 'C#','D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
LIST_NOTES_REST = ['C', 'C#','D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'Rest']

PAIR_NOTES = ['C#', 'D-', 'D#', 'E-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-' ]


NAWBA_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
NAWBA_LIST_TRAIN = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]

TAB_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
TAB_LIST_TRAIN = [1, 2, 3, 5, 9, 10, 14, 16, 18]

MIZAN_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
MIZAN_LIST_TRAIN = [1, 2, 3, 5, 11]

FORM_LIST_TRAIN = [1, 2, 17, 18, 19, 20, 21, 23, 24, 26, 31]

DURATION_BINS= ["1/8", "1/4", "1/2", "3/4", "1", "1.25", "1.5", "2", "3", "4"]

LABEL_LIST_TRAIN = {
    'nawba' : NAWBA_LIST_TRAIN,
    'tab' : TAB_LIST_TRAIN,
    'mizan': MIZAN_LIST_TRAIN,
    'form' : FORM_LIST_TRAIN
}

# Mapping of note names to MIDI numbers
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11, 
            'D-': 1,         'E-': 3,                 'G-': 6,'F##': 7, 'A-': 8,        'B-': 10
}

DISTANCE_MEASURES = ["L1", "L2", "correlation", "camberra", "cosine", "WL2"]

DF_PATH_TRACKS = 'corpus-dataset/documents/'

DF_PATH = 'corpus-dataset/'