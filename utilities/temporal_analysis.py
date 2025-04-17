from music21 import *
import numpy as np
from utilities.model_matching_computation import gaussian, get_distance, train_test_split
from utilities.corpus_search import *
import os
from collections import Counter
from utilities.constants import *
from utilities.temporal_analysis import *
from utilities.model_matching_computation import *
import pandas as pd
from music21 import *
from utilities.corpus_search import *


def get_timestamps_from_all_corpus():
    """
    Parses the musicXML of every track in the dataset into a dataframe of notes, each labeled with all possible target labels, their corresponding section id and
    the following features ['NoteAndRest', 'timestamp_(scs)', 'measure_number', 'quarterLength', 'bpm', 'beatStrength', 'offset', 'tie', 'isChord' ]
    """
    tracks_id = get_all_id_tracks()
    df_sections = parse_all_tracks_sections()
    start_time_seconds = []
    for section in df_sections['end_time']:
        tn = section.split(':')
        start_time_seconds.append((int(tn[0]) * 3600) + (int(tn[1]) * 60) + int(tn[2]))
        
    
    df_sections['end_time_secs'] = start_time_seconds 
    for column in ['nawba', 'tab', 'form', 'mizan']:
        
        df_sections[column] = df_sections[column].apply(
            lambda x: x['id'] if isinstance(x, dict) and 'id' in x else None
        )

    
    #no_tempo_tracks = ['b3d92934-0946-4f2d-8183-312450d7e45e', '719a2afc-461f-461e-ad18-8bce2c4f5023', '99c711e8-0683-4a44-9116-fc2b9448d98d']
    df_labeled_notes = pd.DataFrame()
    for num, track in enumerate(tracks_id):
        path = DF_PATH_TRACKS + track + '/' + track + '.xml'
        if not os.path.exists(path):  continue
        c = converter.parse(path)
        metronome = c.flatten().getElementsByClass(tempo.MetronomeMark)
        bpm = metronome[0].number if len(metronome) > 0 else 80
        seconds_per_beat = 60 / bpm
        time_signature = c.flatten().getElementsByClass('TimeSignature')
        beats_per_measure = time_signature[0].numerator if time_signature else 4

        current_time = 0
        
        timestamps = []
        for part in c.parts:
            for measure in part.getElementsByClass('Measure'):
                measure_start_time = current_time

                for element in measure.flatten():
                    if isinstance(element, tempo.MetronomeMark) and element.number:
                        bpm = element.number
                        seconds_per_beat = 60 / bpm

                if measure.timeSignature is not None:
                    beats_per_measure = measure.timeSignature.numerator

                for element in measure.flatten().notesAndRests:
                    note_offset = element.offset
                    note_duration = element.duration.quarterLength
                    note_time = round(measure_start_time + (note_offset * seconds_per_beat), 2)
                    
                    if isinstance(element, note.Rest):
                        print(element.isChord)
                        timestamps.append((f"Rest", note_time, element.measureNumber, element.duration.quarterLength, bpm, element.beatStrength, element.offset, element.tie, element.isChord))
                    elif isinstance(element, note.Note):
                        timestamps.append((element.nameWithOctave, note_time, element.measureNumber, element.duration.quarterLength, bpm, element.beatStrength, element.offset, element.tie, element.isChord))


                current_time += measure.barDuration.quarterLength * seconds_per_beat
        
        df_track = df_sections[df_sections['track_id'] == track]
        index = 0
        mizan_label = []
        form_label = []
        nawba_label = []
        tab_label = []
        section_id = []
        for note_name, note_time, measure, ql, bpm, beatStrength, offset, tie, isChord in timestamps:
            
            if index > len(df_track) - 1:
                    mizan_label.append(None)
                    form_label.append(None)
                    nawba_label.append(None)
                    tab_label.append(None)
                    section_id.append(track + str(index))
            elif not (note_time <= df_track['end_time_secs'].iloc[index]):
                index += 1
                if index > len(df_track) - 1:
                    mizan_label.append(None)
                    form_label.append(None)
                    nawba_label.append(None)
                    tab_label.append(None)
                    section_id.append(track + str(index))
                else:
                    section_id.append(track + str(index))
                    nawba_label.append(df_track['nawba'].iloc[index])
                    tab_label.append(df_track['tab'].iloc[index])
                    mizan_label.append(df_track['mizan'].iloc[index])
                    form_label.append(df_track['form'].iloc[index])
            else: 
                section_id.append(track + str(index))
                nawba_label.append(df_track['nawba'].iloc[index])
                tab_label.append(df_track['tab'].iloc[index])
                mizan_label.append(df_track['mizan'].iloc[index])
                form_label.append(df_track['form'].iloc[index])
        df_tn = pd.DataFrame(timestamps, columns=['NoteAndRest', 'timestamp_(scs)', 'measure_number', 'quarterLength', 'bpm', 'beatStrength', 'offset', 'tie', 'isChord' ])
        
        df_tn['mizan'] = mizan_label
        df_tn['form'] = form_label
        df_tn['nawba'] = nawba_label
        df_tn['tab'] = tab_label
        df_tn['section_id'] = section_id
        df_tn['key'] = len(tab_label) * [c.analyze('key')]
        df_labeled_notes = pd.concat([df_labeled_notes, df_tn])   
    return df_labeled_notes


def get_timestamps_from_corpus(c):
    """
    Parameters:
        c: music21 parsed XML track
    -------------------------------------------------------------------------------------
    Returns a pandas Dataframe with the following features coresponding to the score passed as parameter.
    Features ['NoteAndRest', 'timestamp_(scs)', 'measure_number', 'quarterLength', 'bpm', 'beatStrength', 'offset', 'tie', 'isChord' ]
    """
    metronome = c.flatten().getElementsByClass(tempo.MetronomeMark)
    bpm = metronome[0].number if metronome else 80
    seconds_per_beat = 60 / bpm
    time_signature = c.flatten().getElementsByClass('TimeSignature')
    beats_per_measure = time_signature[0].numerator if time_signature else 4

    current_time = 0
    timestamps = []

    for part in c.parts:
        for measure in part.getElementsByClass('Measure'):
            measure_start_time = current_time

            for element in measure.flatten():
                if isinstance(element, tempo.MetronomeMark):
                    bpm = element.number
                    seconds_per_beat = 60 / bpm

            if measure.timeSignature is not None:
                beats_per_measure = measure.timeSignature.numerator

            for element in measure.flatten().notesAndRests:
                note_offset = element.offset
                note_duration = element.duration.quarterLength
                note_time = round(measure_start_time + (note_offset * seconds_per_beat), 2)
                
                if isinstance(element, note.Rest):
                    timestamps.append((f"Rest ({note_duration} beats)", note_time, element.measureNumber, element.duration.quarterLength))
                elif isinstance(element, note.Note):
                    timestamps.append((element.nameWithOctave, note_time, element.measureNumber, element.duration.quarterLength))


            current_time += measure.barDuration.quarterLength * seconds_per_beat
    total_time = 0
    for event, time, measure_number, duration in timestamps:
        total_time = time

    #print(f"Total duration based on MusicXML: {total_time:.2f} sec \t {total_time / 60:.2f} minutes")
    return timestamps


def convert_folded_scores_in_models_rythm(y_list, std):
    """
    Parameters:
        y_list: list of track ids to compute the rhythm distribution
        std: std. deviation for gaussian smoothening
    -------------------------------------------------------------------------

    Computes rhythm distribiution model
    """
    
    min_bound = -50
    max_bound = 950
    num_bins = 100
    x_distribution = np.linspace(min_bound, max_bound, num_bins)
    y_distribution_list = list()

    for i in range(len(y_list)):
        y_temp = [0] * len(x_distribution)
        for j in range(len(y_list[i])):
            c = std   # standard deviation
            a = y_list[i][j]/ np.sqrt(2 * np.pi * np.power(c, 2.))  # curve height
            b = j * 100  # center of the curve
            y_temp += a * gaussian(x_distribution, b, c)
        tot_curve = sum(y_temp)
        y_temp[:] = [y / tot_curve for y in y_temp]
        y_distribution_list.append(y_temp)

    return x_distribution, y_distribution_list

def get_folded_rhythm_histogram(ql):
    """
    Parameters:
        ql: quarter length list to insert into histogram
    -------------------------------------------------------------------------

    Computes rhythm distribiution histogram and normalizes it
    """
        
    ql = [0.33 if i == '1/3' else i for i in ql]

    # Define finer bins based on DURATION_BINS you're interested in (1/32, 1/16, etc.)
    hist = Counter()
    hist_y = None

    # Bin the quarter lengths
    for duration in ql:
        if duration <= 0.125:
            hist["1/8"] += 1
        elif duration <= 0.25:
            hist["1/4"] += 1
        elif duration <= 0.5:
            hist["1/2"] += 1
        elif duration <= 0.75:
            hist["3/4"] += 1
        elif duration <= 1:
            hist["1"] += 1
        elif duration <= 1.25:
            hist["1.25"] += 1
        elif duration <= 1.5:
            hist["1.5"] += 1
        elif duration <= 2:
            hist["2"] += 1
        elif duration <= 3:
            hist["3"] += 1
        else: 
            hist["4"] += 1
        

    hist_y = [0] * len(DURATION_BINS)
    for i in range(len(DURATION_BINS)):
        hist_y[i] = hist[DURATION_BINS[i]]
    tot_y_temp = sum(hist_y)
    hist_y[:] = [y / tot_y_temp for y in hist_y]
    return hist_y

def skf_template_rhythm_matching(df, df_notes, std, split, rn):
    """
    Parameters:
        df: dataframe with section id and target label
        df_notes: General note json stored in note_corpus{i}.json
        std: standard deviation value to compute gaussina smoothening with
        split: how many folds to perform
    -------------------------------------------------------------------

    Perfoms the model matching experiment by splitting the dataset into sections of tracks.
    Computes histogram into models and then compares it to each comprehensive target label model.
    """
    X = df.filter(['section_id', 'mizan'])
    y = df.filter(['mizan'])
    mizan_models = []
    overall_acc = []
    actual = []
    predicted = []
    
    for i in range(split):
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state= i * rn)
    
        for mizan in MIZAN_LIST_TRAIN:
            ql = df_notes[(df_notes['mizan'] == mizan) & 
                                    (df_notes['section_id'].isin(X_train['section_id'].tolist()))
                                    ]['quarterLength'].tolist()
            
            hist_y_avg = get_folded_rhythm_histogram(ql)
            _, y_model = convert_folded_scores_in_models_rythm([hist_y_avg], std)
            mizan_models.append(y_model[0])
            
            test_models = []
            curr_acc = []
        for j in range(len(X_test)):    
            ql = df_notes[df_notes['section_id'] == X_test['section_id'].iloc[j]]['quarterLength'].tolist()
            hist_y_avg = get_folded_rhythm_histogram(ql)
            
            _, y_model_test = convert_folded_scores_in_models_rythm([hist_y_avg], std)
            test_models.append(y_model_test[0])
            label_score = []
            for l, label_t2 in enumerate(MIZAN_LIST_TRAIN):
                label_score.append((get_distance(y_model_test[0], mizan_models[l], 'L2'), label_t2))
            curr_acc.append(y_test.iloc[j]['mizan'] == min(label_score)[1])
            
            actual.append(y_test.iloc[j]['mizan'])
            predicted.append(min(label_score)[1])
        overall_acc.append(sum(curr_acc) / len(curr_acc))
        
    return actual, predicted, overall_acc
