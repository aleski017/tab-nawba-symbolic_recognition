# Histogram bins
LIST_NOTES = ['C', 'C#','D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
LIST_NOTES_REST = ['C', 'C#','D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'Rest']
PAIR_NOTES = ['C#', 'D-', 'D#', 'E-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-' ]
DURATION_BINS= ["1/8", "1/4", "1/2", "3/4", "1", "1.25", "1.5", "2", "3", "4"]

# Train classes are those labels with enough samples to train
NAWBA_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
TAB_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
NAWBA_LIST_TRAIN = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]
TAB_LIST_TRAIN = [1, 2, 3, 5, 9, 10, 14, 16, 18]
MIZAN_LIST_TRAIN = [1, 2, 3, 5, 11]
MIZAN_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
FORM_LIST_TRAIN = [1, 2, 17, 18, 19, 20, 21, 23, 24, 26, 31]
LABEL_LIST_TRAIN = {
    'nawba' : NAWBA_LIST_TRAIN,
    'tab' : TAB_LIST_TRAIN,
    'mizan': MIZAN_LIST_TRAIN,
    'form' : FORM_LIST_TRAIN
}

LABEL_LIST = {
    'nawba' : NAWBA_LIST,
    'tab' : TAB_LIST,
    'mizan': MIZAN_LIST
}

# Mappings
REPLACE_STRING = {
    'nawba' : {'1' : 0, '2' :1, '3' :2, '4' :3, '5' :4, '6' :5, '7' :6, '10' :7, '11' :8, '12' :9, '13':10},
    'tab' : {'1':0, '2':1, '3':2, '5':3, '9':4, '10':5, '14':6, '16':7, '18':8},
    'mizan' : {'1':0, '2':1, '3':2, '5':3, '11':4}
}
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11, 
            'D-': 1,         'E-': 3,                 'G-': 6,'F##': 7, 'A-': 8,        'B-': 10
}

# Corpus search
DF_PATH_TRACKS = 'corpus-dataset/documents/'
PREFIX_LENGTH = len('f461045b-50bc-4b20-a731-66fbd3a264ae')
DF_PATH = 'corpus-dataset/'

DISTANCE_MEASURES = ["L1", "L2", "correlation", "camberra", "cosine", "WL2"]