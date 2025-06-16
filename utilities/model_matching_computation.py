import numpy as np
from utilities.constants import *
from music21 import converter
import os
from utilities.corpus_search import *
from scipy.spatial import distance
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re
from sklearn.model_selection import train_test_split



def gaussian(x, mu, sig):
    """
    Helper function to smoothen histograms
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def convert_folded_scores_in_models(y_list, std):
    """
    Parameters
        y_list: A list of pitch distribution histograms
        std: Standard deviation for gaussian smoothening
    ---------------------------------------------------------------------------

    For each histogram given as input, a smoothened function model is in output
    """
    min_bound = -50
    max_bound = 1150
    num_bins = 1200
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
        if tot_curve != 0:
            y_temp[:] = [y / tot_curve for y in y_temp]
        y_distribution_list.append(y_temp)

    return x_distribution, y_distribution_list


def get_folded_score_histogram(tracks):

    """
    Parameters
        tracks: A list of track_ids to compute a comprehensive histogram
    ---------------------------------------------------------------------------

    Returns a pitch distribution histogram of all notes in each track given in input 
    """
    hist = dict((note, 0) for note in LIST_NOTES)
    hist_y = None
    for track_id in tracks:
        
        path = DF_PATH_TRACKS + track_id + '/' + track_id + '.xml'
        if os.path.exists(path):
            rmbid_parsing= converter.parse(path)
            #rmbid_parsing = converter.parse(os.path.join(recordings_folder, rmbid, rmbid) + XML_SUFFIX)
            rmbid_stream = rmbid_parsing.recurse().notes
            for myNote in rmbid_stream:
                
                if myNote.isChord:
                    print(str(myNote) + " discarded")
                else:
                    if myNote.name in LIST_NOTES:
                        hist[myNote.name] = hist[myNote.name] + myNote.duration.quarterLength
                    else:
                        if myNote.name in PAIR_NOTES:
                            i = PAIR_NOTES.index(myNote.name)
                            name_translated = PAIR_NOTES[i - 1]
                            hist[name_translated] = hist[name_translated] + myNote.duration.quarterLength
                        else:
                            print(str(myNote) + " discarded")
                            # print(rmbid)
            
            hist_y = [0] * len(LIST_NOTES)
            for i in range(len(LIST_NOTES)):
                hist_y[i] = hist[LIST_NOTES[i]]
            
    return hist_y


def compute_avg_folded_hist_scores_by_nawba():
    """
    Returns a folded(octave is not take into account) histogram for each nawba class
    """
    hist_y_avg = []
    for nawba in NAWBA_LIST:
        print(f"Computing for nawba {nawba}")
        y_temp = get_folded_score_histogram(get_id_track_by_nawba(nawba))
        tot_y_temp = sum(y_temp)
        y_temp[:] = [y / tot_y_temp for y in y_temp]
        hist_y_avg.append(y_temp)
    return LIST_NOTES, hist_y_avg


def compute_avg_folded_hist_scores(tracks_to_avg):
    """
    Parameters:
        tracks_to_avg: list of track ids to compute the pitch distribution
    -------------------------------------------------------------------------

    Computes pitch distribiution histogram and normalizes it
    """
    y_temp = get_folded_score_histogram(tracks_to_avg)
    tot_y_temp = sum(y_temp)
    y_temp[:] = [y / tot_y_temp for y in y_temp]
    hist_y_avg = y_temp
    
    return LIST_NOTES, hist_y_avg


def get_distance(shifted_recording, nawba_model, distance_type):
    """
    Parameters:
        shifted_recording: test sample to compare with model
        nawba_model: model of a class to compare with shifted recording
        distance_type: Measure to compute distance between the models
    -------------------------------------------------------------------

    Returns a numeric value distance between two gaussain models
    """

    # city-block (L1)
    if distance_type == DISTANCE_MEASURES[0]:
        return distance.cityblock(shifted_recording, nawba_model)
    # euclidian (L2)
    if distance_type == DISTANCE_MEASURES[1]:
        return distance.euclidean(shifted_recording, nawba_model)
    # correlation
    if distance_type == DISTANCE_MEASURES[2]:
        return 1/np.dot(shifted_recording, nawba_model)
    
    # camberra
    if distance_type == DISTANCE_MEASURES[3]:
        return distance.canberra(shifted_recording, nawba_model)
    # cosine
    if distance_type == DISTANCE_MEASURES[4]:
        return distance.cosine(shifted_recording, nawba_model)
    

def create_gaussian_templates(track_list, std):
    """
    Parameters:
        track_list: list of track to compute the models with
        std: standard deviation value to compute gaussina smoothening with
    -------------------------------------------------------------------

    Returns a numeric value distance between two gaussain models
    """
    _, y_temp = compute_avg_folded_hist_scores(track_list)        
    _, y_model = convert_folded_scores_in_models([y_temp], std)
    return y_model


def print_performance(actual, predicted, overall_acc=None):
    """
    Parameters:
        actual: true target label
        predicted: list of labels returned by the model
        overall_acc: list of accuracy values over different runs
    -------------------------------------------------------------------

    Prints different metrics:
        Precision;
        Std. deviation(if overall_acc given);
        Weighted recall;
        Global recall; 
        F1 Score;
        Confusion matrix.
    """

    print(f"Precision Micro: {precision_score(actual, predicted, average='micro'):4f}", flush=True)
    print(f"Precision Macro: {precision_score(actual, predicted, average='macro'):4f}", flush=True)
    print(f"Precision Weighted: {precision_score(actual, predicted, average='weighted'):4f}", flush=True)
    if overall_acc is not None:
        print(f"Std. Deviation: {np.std(overall_acc, ddof=1):.4f}")
    print(f"Recall Macro: {recall_score(actual, predicted, average='macro'):4f}", flush=True)
    print(f"Recall Micro: {recall_score(actual, predicted, average='micro'):4f}", flush=True)
    print(f"Recall Weighted: {recall_score(actual, predicted, average='weighted'):4f}", flush=True)
    print(f"F1 score Weighted: {f1_score(actual, predicted, average='weighted'):4f}", flush=True)
    print(f"F1 score Macro: {f1_score(actual, predicted, average='macro'):4f}", flush=True)

    print(f"F1 score Micro: {f1_score(actual, predicted, average='micro'):4f}", flush=True)

    cm = confusion_matrix(actual, predicted, labels = list(set(actual)))
    disp = ConfusionMatrixDisplay(cm, display_labels= list(set(actual)))
    disp.plot()


def skf_model_matching_label(df, df_notes, std, split, label):
    """
    Parameters:
        df: dataframe with section id and target label
        df_notes: General note json stored in note_corpus{i}.json
        std: standard deviation value to compute gaussina smoothening with
        label: target label to perform the experiment on
    -------------------------------------------------------------------

    Perfoms the model matching experiment by splitting the dataset into sections of tracks.
    Computes histogram into models and then compares it to each comprehensive target label model.
    """
    X = df.filter(['section_id', label])
    y = df.filter([label])
    df_notes['NoteAndRest'] = df_notes['NoteAndRest'].apply(lambda x: re.sub(r'\d+', '', x))

    label_models = []
    overall_acc = []
    actual = []
    predicted = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=4)
    
    for label in LABEL_LIST_TRAIN[label]:
        nr = df_notes[
                                (df_notes[label] == label) & 
                                (df_notes['section_id'].isin(X_train['section_id'].tolist()))
                                ]['NoteAndRest'].tolist()
        ql = df_notes[(df_notes[label] == label) & 
                                (df_notes['section_id'].isin(X_train['section_id'].tolist()))
                                ]['quarterLength'].tolist()
        ql = [0.33 if i == '1/3' else i for i in ql]

        hist = dict((note, 0) for note in LIST_NOTES)
        hist_y = None
        for i, myNote in enumerate(nr):
                if myNote in LIST_NOTES:
                    hist[myNote] = hist[myNote] + ql[i]
                else:
                    if myNote in PAIR_NOTES:
                        i = PAIR_NOTES.index(myNote)
                        name_translated = PAIR_NOTES[i - 1]
                        hist[name_translated] = hist[name_translated] + ql[i]
        hist_y = [0] * len(LIST_NOTES)
        for i in range(len(LIST_NOTES)):
            hist_y[i] = hist[LIST_NOTES[i]]
        tot_y_temp = sum(hist_y)
        hist_y[:] = [y / tot_y_temp for y in hist_y]
        hist_y_avg = hist_y
        _, y_model = convert_folded_scores_in_models([hist_y_avg], std)
        label_models.append(y_model[0])

    test_models = []
    curr_acc = []
    for j in range(len(X_test)):
        nr = df_notes[
                                df_notes['section_id'] == X_test['section_id'].iloc[j]
                                ]['NoteAndRest'].tolist()
        
        ql = df_notes[df_notes['section_id'] == X_test['section_id'].iloc[j]]['quarterLength'].tolist()
        ql = [0.33 if i == '1/3' else i for i in ql]
        
        hist = dict((note, 0) for note in LIST_NOTES)
        hist_y = None
        for i, myNote in enumerate(nr):
                if myNote in LIST_NOTES:
                    hist[myNote] = hist[myNote] + ql[i]
                else:
                    if myNote in PAIR_NOTES:
                        i = PAIR_NOTES.index(myNote)
                        name_translated = PAIR_NOTES[i - 1]
                        hist[name_translated] = hist[name_translated] + ql[i]
                
        hist_y = [0] * len(LIST_NOTES)
        for i in range(len(LIST_NOTES)):
            hist_y[i] = hist[LIST_NOTES[i]]
        tot_y_temp = sum(hist_y)
        if tot_y_temp != 0: 
            hist_y[:] = [y / tot_y_temp for y in hist_y]
        hist_y_avg = hist_y
        _, y_model_test = convert_folded_scores_in_models([hist_y_avg], std)
        
        #print(X_test.iloc[j]['section_id'])
        test_models.append(y_model_test[0])
        label_score = []
        for l, label_t2 in enumerate(LABEL_LIST_TRAIN[label]):
            label_score.append((get_distance(y_model_test[0], label_models[l], 'L2'), label_t2))
        
        
        curr_acc.append(y_test.iloc[j][label] == min(label_score)[1])
        actual.append(y_test.iloc[j][label])
        predicted.append(min(label_score)[1])
    overall_acc.append(sum(curr_acc) / len(curr_acc))
    return actual, predicted, overall_acc


def skf_model_matching(label, df, random_state, k, test_dim, std):
    """
    Parameters:
        df: dataframe with track id and target label
        std: standard deviation value to compute gaussina smoothening with
        label: target label to perform the experiment on
        test_dim: Number of samples to take in the test set
    -------------------------------------------------------------------

    Perfoms the model matching experiment by gathering tracks and computing the model of each label.
    Computes histogram into models and then compares it to each comprehensive target label model.
    """
    # PREPARE DATASET FOR TRAINING
    X = df.filter(['track_id', label])
    y = df.filter([label])

    overall_acc= []
    actual = []
    predicted = []
    y_model_list = []
    # Stratified K-Fold Cross-Validation LOOP
    for fold in range(k):        
        test_indices = []
        train_indices = []

        for label_t in y[label].unique():
            label_indices = y[y[label] == label_t].index.to_list()
        
            # SET REPRODUCIBLE SEED FOR SHUFFLE
            np.random.seed(fold * random_state)  
            np.random.shuffle(label_indices)

            test_indices.extend(label_indices[:test_dim])  
            train_indices.extend(label_indices[test_dim:])  
            

        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        _, y_test = y.loc[train_indices], y.loc[test_indices]
        
        # CREATE TEMPLATES FOR EACH LABEL
        y_model_list = []
        for label_t2 in LABEL_LIST_TRAIN[label]:
            _, y_temp = compute_avg_folded_hist_scores(X_train.loc[X_train[label] == label_t2, 'track_id'].tolist())        
            _, y_model = convert_folded_scores_in_models([y_temp], std)
            y_model_list.append(y_model[0])

        # COMPUTE MODEL OF TEST SAMPLES
        curr_acc = []
        for i in range(len(X_test)):
            _, y_temp = compute_avg_folded_hist_scores([X_test.iloc[i]['track_id']])
            _, y_model_list_test = convert_folded_scores_in_models([y_temp], std)

            # RUN ACCURACY TEST
            label_score = []
            for l, label_t3 in enumerate(LABEL_LIST_TRAIN[label]):
                label_score.append((get_distance(y_model_list_test[0], y_model_list[l], 'L2'), label_t3))
            
            curr_acc.append(y_test.iloc[i][label] == min(label_score)[1])
            actual.append(y_test.iloc[i][label])
            predicted.append(min(label_score)[1])
        overall_acc.append(sum(curr_acc) / len(curr_acc))
        
    return actual, predicted, overall_acc  

def compute_avg_folded_hist_labeled_notes_models(nr, ql, std):
    """
    Parameters:
        nr: List of note to use for pitch distribution histogram
        ql: List of quarter lengths of notes to use for rhythm histogram
        std: std. deviation to compute for gaussian smoothening
    -------------------------------------------------------------------

    Return the function models of the given slice of notes
    """
    hist_y_avg = compute_avg_folded_hist_labeled_notes(nr, ql)
    _, y_model_test = convert_folded_scores_in_models([hist_y_avg], std)
    return y_model_test

def compute_avg_folded_hist_labeled_notes(nr, ql):
    """
    Parameters:
        nr: List of note to use for pitch distribution histogram
        ql: List of quarter lengths of notes to use for rhythm histogram
    -------------------------------------------------------------------

    Return the histogram pitch distribution of the given slice of notes
    """
    hist = dict((note, 0) for note in LIST_NOTES)
    hist_y = None
    for i, myNote in enumerate(nr):
            if myNote in LIST_NOTES:
                hist[myNote] = hist[myNote] + ql[i]
    
            else:
                if myNote in PAIR_NOTES:
                    i = PAIR_NOTES.index(myNote)
                    name_translated = PAIR_NOTES[i - 1]
                    hist[name_translated] = hist[name_translated] + ql[i]
            
    hist_y = [0] * len(LIST_NOTES)
    for i in range(len(LIST_NOTES)):
        hist_y[i] = hist[LIST_NOTES[i]]
    tot_y_temp = sum(hist_y)
    if tot_y_temp != 0: 
        hist_y[:] = [y / tot_y_temp for y in hist_y]
    return hist_y
 

def label_length(df_notes):
    """
    Parameters:
        df_notes: Dataframe of note you want to add the length of the section
    -------------------------------------------------------------------

    Computes the lenght of the section given the first and last timestamp of the section and adds the new column
    """
    if 'length_section' in df_notes.columns:
        df_notes.drop('length_section', axis=1, inplace=True)

    # Create a function to compute the section length
    def compute_section_length(df):
        # Group by section_id and calculate the section length: max - min timestamp
        section_lengths = df.groupby('section_id')['timestamp_(scs)'].agg(['min', 'max'])
        section_lengths['length_section'] = section_lengths['max'] - section_lengths['min']
        
        # Return the section length data
        return section_lengths[['length_section']]

    # Now calculate the section length for each section and merge back with the original DataFrame
    section_lengths = compute_section_length(df_notes)

    # Merge the section lengths back to the original dataframe based on 'section_id'
    df_notes = df_notes.merge(section_lengths, on='section_id', how='left')
    return df_notes

"""
def skf_model_matching_label(df, df_notes, std, k, test_dim, label, random_state):
    X = df.filter(['section_id', label])
    y = df.filter([label])
    df_notes['NoteAndRest'] = df_notes['NoteAndRest'].apply(lambda x: re.sub(r'\d+', '', x))

    label_models = []
    overall_acc = []
    actual = []
    predicted = []
    
    for q in range(1, k):
        
        test_indices = []
        train_indices = []
        
        for label_t in y[label].unique():
            label_indices = y[y[label] == label_t].index.to_list()

            # Set reproducible seed for shuffle
            np.random.seed(q * random_state)  
            np.random.shuffle(label_indices)

            test_indices.extend(label_indices[:test_dim])  
            train_indices.extend(label_indices[test_dim:]) 
             
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        _, y_test = y.loc[train_indices], y.loc[test_indices]
        
        
        #X_train, X_test, _, y_test = train_test_split(X, y, test_size=test_dim, stratify=y, random_state=random_state * q)

        for label_id in LABEL_LIST_TRAIN[label]:
            nr = df_notes[
                (df_notes[label] == label_id) & 
                (df_notes['section_id'].isin(X_train['section_id'].tolist()))
            ]['NoteAndRest'].tolist()
            ql = df_notes[
                (df_notes[label] == label_id) & 
                (df_notes['section_id'].isin(X_train['section_id'].tolist()))
            ]['quarterLength'].tolist()
            ql = [0.33 if i == '1/3' else i for i in ql]

            y_model = compute_avg_folded_hist_labeled_notes_models(nr, ql, std)
            label_models.append(y_model[0])

        test_models = []
        curr_acc = []

        for j in range(len(X_test)):
            
            section_id = X_test['section_id'].iloc[j]

            nr = df_notes[df_notes['section_id'] == section_id]['NoteAndRest'].tolist()
            ql = df_notes[df_notes['section_id'] == section_id]['quarterLength'].tolist()
            ql = [0.33 if i == '1/3' else i for i in ql]
            # Collect neighboring sections' notes
            nr_neighbors = nr
            ql_neighbors = ql

            
            # Compute the model for the sampled window of notes
            y_model_test = compute_avg_folded_hist_labeled_notes_models(nr_neighbors, ql_neighbors, std)
            test_models.append(y_model_test[0])


            # Compare with label models using distance (e.g., L2 distance)
            label_score = []
            for l, label_t2 in enumerate(LABEL_LIST_TRAIN[label]):
                label_score.append((get_distance(y_model_test[0], label_models[l], 'L2'), label_t2))

            # Find the best matching label (minimum distance)
            predicted_label = min(label_score)[1]
            #print(f"True label: {y_test.iloc[j][label]}, Predicted label: {predicted_label}")
            curr_acc.append(y_test.iloc[j][label] == predicted_label)
            actual.append(y_test.iloc[j][label])
            predicted.append(predicted_label)

        overall_acc.append(sum(curr_acc) / len(curr_acc))

    return actual, predicted, overall_acc
"""

def merge_sections(df, label,THRESHOLD = 600):
    """
    Parameters:
        df: Dataframe to merge
        label: Target label to merge on
        THRESHOLD: sections under this threshold will be merged into other of the same label
    -------------------------------------------------------------------

    Merges sections under the given threshold with sections of the same label
    """
    df = label_length(df)
    df_under = df[df['length_section'] < THRESHOLD]['section_id'].unique()
    prev = 0
    while len(df_under) != prev:
        
        df_done = []
        prev = len(df_under)
        for track in df_under:
            if track in df_done: continue
            target_label = df[df['section_id'] == track][label].iloc[0]
            if len(df[(df['section_id'].isin(df_under))\
                                            & (df[label] == target_label)                                           
                                            & (df['section_id'] != track)]['section_id']) == 0: continue
            
            to_merge = (df[(df['section_id'].isin(df_under))\
                                            & (df[label] == target_label)                                        
                                            
                                            & (df['section_id'] != track)]['section_id'].iloc[0])
            
            df.loc[(df['section_id'] == track) |
                                    (df['section_id'] == to_merge) , 'section_id'] = track + to_merge
            df_done.append(track)
            df_done.append(to_merge)
        
        df = label_length(df)
        df_under = df[df['length_section'] < THRESHOLD]['section_id'].unique()
        

    secs_under = (df[(df['length_section'] < THRESHOLD)]['section_id'].unique())

    for sec in secs_under:
        
        target_label = df[df['section_id'] == sec][label].iloc[0]
        
        if len(df[ (df[label] == target_label)
                                        #& (df['length_section'] < THRESHOLD + 200)   
                                        & (df['section_id'] != sec)]['section_id']) == 0: continue
        
        to_merge = (df[ (df[label] == target_label)
                                        #& (df['length_section'] < THRESHOLD + 200)   
                                        & (df['section_id'] != sec)]['section_id'].iloc[0])
        
        df.loc[(df['section_id'] == sec) |
                                (df['section_id'] == to_merge) , 'section_id'] = sec + to_merge
        
        df = label_length(df)
        #if sec in df_done: continue
    return df
