from utilities.constants import *
from utilities.temporal_analysis import *
import matplotlib.pyplot as plt
from utilities.model_matching_computation import *
from utilities.features_eng import *
import pandas as pd
from music21 import *
from utilities.corpus_search import *
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from collections import defaultdict

import warnings
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import torch
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.nn.utils as rnn_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.utils.class_weight import compute_class_weight
from utilities.dl_utilities import *
from graphmuse.loader import MuseNeighborLoader
warnings.filterwarnings("ignore")
from sklearn import svm




def set_fixed_seed_dl(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_test_val_split(X, y, random_state):

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, stratify=y_train_temp, test_size=0.11, random_state=random_state)
    return X_train, X_test, X_val, y_train, y_test, y_val

def load_graphs_perLabel(label):
    graphs = list()
    graphs_y = list()
    for track in get_all_id_tracks():

        path = DF_PATH_TRACKS + track + '/' + track + '.xml'
        if not os.path.exists(path): continue
        graph = torch.load(f'graphs/{track}.pt')

        if not graph['note'][f'y_{label}'] in (LABEL_LIST_TRAIN[label]): continue
        graphs_y.append(REPLACE_STRING[label][str(int(graph['note'][f'y_{label}']))])
        graphs.append(graph)
    return graphs, graphs_y

def run_ModelBasedAI_experiment(label):
    # PREPARE DATASET FOR TRAINING 
    all_tracks_id = []
    for track in get_all_id_tracks():
        path = DF_PATH_TRACKS + track + '/' + track + '.xml'
        if os.path.exists(path):
                all_tracks_id.append(track)
    df = pd.DataFrame({
        'track_id': all_tracks_id,
        label: get_id_label(label)
        })

    df = df[df[label].isin(LABEL_LIST_TRAIN[label])]
    #df.info()
    actual = []
    pred = []
    overall_acc = []

    y_test, y_pred, overall_acc = skf_model_matching(label, df, random_state= 42, k= 10, std=30)
    print_performance(y_test, y_pred, overall_acc)

def run_GaussianNB_experiment(label, overlap, sequence_length, random_state, folds = 10, print_results = True):
    print(f"For [{label}]: Stride {overlap}% || Length of {sequence_length} notes", flush=True)
    skf = StratifiedKFold(n_splits=folds)
    df, pitch_distr_sections, ql_distr_sections = prepare_dataframe(label, overlap, sequence_length)
    prefixes, prefixes_y, prefix_to_ids = return_prefixes(df, label)
    df_windowed = normalize_and_traspose(df, label, pitch_distr_sections, ql_distr_sections)
    prefixes = np.array(prefixes)
    actual = []
    pred = []
    overall_acc = []
    print(f"Done with fold: ", end="", flush=True)
    for i, (train_index, test_index) in enumerate(skf.split(prefixes, prefixes_y)):
        print(f" {i},", end="", flush=True)
        #train_prefixes, test_prefixes = train_test_split(prefixes, stratify=prefixes_y, test_size=0.2, random_state=random_state)

        train_prefixes = prefixes[train_index]
        test_prefixes= prefixes[test_index]

        # Step 3: Gather IDs for train and test sets
        train_ids = [id_ for prefix in train_prefixes for id_ in prefix_to_ids[prefix]]
        test_ids = [id_ for prefix in test_prefixes for id_ in prefix_to_ids[prefix]]

        X_train, y_train = df_windowed[df_windowed['section_id'].isin(train_ids)]['global_features'], df_windowed[df_windowed['section_id'].isin(train_ids)][label].tolist()
        X_test, y_test = df_windowed[df_windowed['section_id'].isin(test_ids)]['global_features'], df_windowed[df_windowed['section_id'].isin(test_ids)][label].tolist()
        # Stack features
        global_train = np.vstack(X_train.to_numpy())
        global_test = np.vstack(X_test.to_numpy())

        # Fit scaler on all columns except last
        scaler = StandardScaler().fit(global_train[:, :-1])

        # Transform and reattach last column
        X_train = list(np.hstack([scaler.transform(global_train[:, :-1]), global_train[:, -1:]]))
        X_test = list(np.hstack([scaler.transform(global_test[:, :-1]), global_test[:, -1:]]))

        class_counts = np.bincount(y_train)
        priors = class_counts / class_counts.sum()

        clf = GaussianNB(priors=priors)
        clf.fit(X_train, y_train)

        actual.extend(y_test)
        pred.extend(clf.predict(X_test))
        overall_acc.append(clf.score(X_test, y_test))

        # Check if it converged and the testing accuracy
        #print("testing accuracy:", clf.score(X_test, y_test), flush=True)
    if print_results:
        print(f"\n {'-' * 20}", flush=True)
        print_performance(actual, pred , overall_acc)

def run_SVC_experiment(label, overlap, sequence_length, random_state, folds = 10, print_results = True):

    print(f"For [{label}]: Stride {overlap}% || Length of {sequence_length} notes", flush=True)
    random.seed(random_state)
    np.random.seed(random_state)
    df, pitch_distr_sections, ql_distr_sections = prepare_dataframe(label, overlap, sequence_length)
    prefixes, prefixes_y, prefix_to_ids = return_prefixes(df, label)
    df_windowed = normalize_and_traspose(df, label, pitch_distr_sections, ql_distr_sections)
    skf = StratifiedKFold(n_splits=folds)
    prefixes = np.array(prefixes)
    actual = []
    pred = []
    rec_w = []
    f1_w= []
    overall_acc = []
    print(f"Done with fold: ", end="", flush=True)
    for i, (train_index, test_index) in enumerate(skf.split(prefixes, prefixes_y)):
        print(f" {i},", end="", flush=True)
        #train_prefixes, test_prefixes = train_test_split(prefixes, stratify=prefixes_y, test_size=0.2, random_state=random_state)

        train_prefixes = prefixes[train_index]
        test_prefixes= prefixes[test_index]

        # Step 3: Gather IDs for train and test sets
        train_ids = [id_ for prefix in train_prefixes for id_ in prefix_to_ids[prefix]]
        test_ids = [id_ for prefix in test_prefixes for id_ in prefix_to_ids[prefix]]


        X_train, y_train = df_windowed[df_windowed['section_id'].isin(train_ids)]['global_features'], df_windowed[df_windowed['section_id'].isin(train_ids)][label].tolist()
        X_test, y_test = df_windowed[df_windowed['section_id'].isin(test_ids)]['global_features'], df_windowed[df_windowed['section_id'].isin(test_ids)][label].tolist()

        # Stack features
        global_train = np.vstack(X_train.to_numpy())
        global_test = np.vstack(X_test.to_numpy())

        # Fit scaler on all columns except last
        scaler = StandardScaler().fit(global_train[:, :-1])

        # Transform and reattach last column
        X_train = list(np.hstack([scaler.transform(global_train[:, :-1]), global_train[:, -1:]]))
        X_test = list(np.hstack([scaler.transform(global_test[:, :-1]), global_test[:, -1:]]))


        clf = svm.SVC(C=1e5, kernel = 'rbf',max_iter=50000, class_weight='balanced')
        clf.fit(X_train, y_train)
        actual.extend(y_test)
        pred.extend(clf.predict(X_test))
        rec_w.append(recall_score(y_test, clf.predict(X_test), average='weighted'))
        f1_w.append(f1_score(y_test, clf.predict(X_test), average='weighted'))
        overall_acc.append(clf.score(X_test, y_test))

        # Check if it converged and the testing accuracy
        #print("testing accuracy:", clf.score(X_test, y_test), flush=True)
    if print_results:
        print(f"\n {'-' * 20}", flush=True)
        print_performance(actual, pred , overall_acc)
        print("Recall_std ", np.array(rec_w).std())
        print("f1_std ", np.array(f1_w).std())
    return actual, pred, overall_acc
def run_RandomForest_experiment(label, overlap, sequence_length, random_state, folds = 10, hyperparameter_tuning = False, print_results = True):
    best_params = None
    skf = StratifiedKFold(n_splits=folds)
    print(f"For [{label}]: Stride {overlap}% || Length of {sequence_length} notes", flush=True)
    pred = []
    actual = []
    rec_w = []
    f1_w= []
    overall_acc = []
    df, pitch_distr_sections, ql_distr_sections = prepare_dataframe(label, overlap, sequence_length)
    prefixes, prefixes_y, prefix_to_ids = return_prefixes(df, label)

    for i, (train_index, test_index) in enumerate(skf.split(prefixes, prefixes_y)):
        print(f" {i},", end="", flush=True )
        random.seed(random_state)
        np.random.seed(random_state)
        #train_prefixes, test_prefixes = train_test_split(prefixes, stratify=prefixes_y, test_size=0.2, random_state=random_state)
        train_prefixes = prefixes[train_index]
        test_prefixes= prefixes[test_index]
        # Step 3: Gather IDs for train and test sets
        train_ids = [id_ for prefix in train_prefixes for id_ in prefix_to_ids[prefix]]

        #compare_plot_label_distribution(df_windowed, df_notes)
        shuffled_df = df.sample(frac=1, random_state=random_state)
        # Group by 'section_id'
        grouped = shuffled_df.groupby("section_id", group_keys=True)[['NoteAndRest', 'quarterLength', 'timestamp_(scs)', 'length_section', 'bpm', 'key', 'offset', 'beatStrength', label]]
        # Create an array for each group

        group_arrays = {section_id: group.values for section_id, group in grouped}
        sequences_features = []
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for key, values in group_arrays.items():
            if key in train_ids:
                X_train.append(extract_feature(values, pitch_distr_sections[key], ql_distr_sections[key]))
                y_train.append((values[0][-1]))
            else:
                X_test.append(extract_feature(values, pitch_distr_sections[key], ql_distr_sections[key]))
                y_test.append((values[0][-1]))

            sequences_features.append(extract_feature(values, pitch_distr_sections[key], ql_distr_sections[key]))


        sequences_features = [list(x.values()) for x in sequences_features]
        X_train = [list(x.values()) for x in X_train]
        X_test = [list(x.values()) for x in X_test]

        if hyperparameter_tuning and i == 0:
            clf = RandomForestClassifier()
            # Initialization of hyperparameter choice
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            max_features = ['sqrt', 'log2', None]
            criterion = ['gini', 'entropy', 'log_loss']
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'criterion': criterion,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}

            rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 10, cv = 10, random_state=random_state, n_jobs = -1)
            rf_random.fit(X_train, y_train)
            best_params = rf_random.best_params_

        if hyperparameter_tuning and i > 0:
            clf = RandomForestClassifier(**best_params)
        if label == "nawba":
            clf = RandomForestClassifier(n_estimators=800, random_state=random_state, min_samples_split=10, min_samples_leaf=1, max_features='sqrt', max_depth = 70, bootstrap=False, criterion= 'gini', class_weight='balanced')
        else:
            clf = RandomForestClassifier(n_estimators=1200, random_state=random_state, min_samples_split=5, min_samples_leaf=2, max_features='log2', max_depth = 110, bootstrap=False, criterion= 'entropy', class_weight='balanced')

        clf.fit(X_train, y_train)
        pred.extend(clf.predict(X_test))
        actual.extend(y_test)
        rec_w.append(recall_score(y_test, clf.predict(X_test), average='weighted'))
        f1_w.append(f1_score(y_test, clf.predict(X_test), average='weighted'))
        overall_acc.append(clf.score(X_test, y_test))

        # Check if it converged and the testing accuracy
        #print("testing accuracy:", clf.score(X_test, y_test), flush=True)
    if print_results:
        print(f"\n {'-' * 20}", flush=True)
        print_performance(actual, pred , overall_acc)
        print("Recall_std ", np.array(rec_w).std())
        print("f1_std ", np.array(f1_w).std())
    return actual, pred, overall_acc
def run_KNN_experiment(label, overlap, sequence_length, random_state, folds = 10, print_results = True):
    print(f"For [{label}]: Stride {overlap}% || Length of {sequence_length} notes", flush=True)
    skf = StratifiedKFold(n_splits=folds)
    df, pitch_distr_sections, ql_distr_sections = prepare_dataframe(label, overlap, sequence_length)
    prefixes, prefixes_y, prefix_to_ids = return_prefixes(df, label)


    df_windowed = normalize_and_traspose(df, label, pitch_distr_sections, ql_distr_sections)
    prefixes = np.array(prefixes)
    actual = []
    rec_w= []
    f1_w= []
    pred = []
    overall_acc = []
    print(f"Done with fold: ", end="", flush=True)
    for i, (train_index, test_index) in enumerate(skf.split(prefixes, prefixes_y)):
        print(f" {i},", end="", flush=True)
        #train_prefixes, test_prefixes = train_test_split(prefixes, stratify=prefixes_y, test_size=0.2, random_state=random_state)

        train_prefixes = prefixes[train_index]
        test_prefixes= prefixes[test_index]

        # Step 3: Gather IDs for train and test sets
        train_ids = [id_ for prefix in train_prefixes for id_ in prefix_to_ids[prefix]]
        test_ids = [id_ for prefix in test_prefixes for id_ in prefix_to_ids[prefix]]


        X_train, y_train = df_windowed[df_windowed['section_id'].isin(train_ids)]['global_features'], df_windowed[df_windowed['section_id'].isin(train_ids)][label].tolist()
        X_test, y_test = df_windowed[df_windowed['section_id'].isin(test_ids)]['global_features'], df_windowed[df_windowed['section_id'].isin(test_ids)][label].tolist()

        # Stack features
        global_train = np.vstack(X_train.to_numpy())
        global_test = np.vstack(X_test.to_numpy())

        # Fit scaler on all columns except last
        scaler = StandardScaler().fit(global_train[:, :-1])

        # Transform and reattach last column
        X_train = list(np.hstack([scaler.transform(global_train[:, :-1]), global_train[:, -1:]]))
        X_test = list(np.hstack([scaler.transform(global_test[:, :-1]), global_test[:, -1:]]))

        class_counts = np.bincount(y_train)
        priors = class_counts / class_counts.sum()

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X_train, y_train)

        actual.extend(y_test)
        pred.extend(clf.predict(X_test))
        rec_w.append(recall_score(y_test, clf.predict(X_test), average='weighted'))
        f1_w.append(f1_score(y_test, clf.predict(X_test), average='weighted'))
        overall_acc.append(clf.score(X_test, y_test))

        # Check if it converged and the testing accuracy
        #print("testing accuracy:", clf.score(X_test, y_test), flush=True)
    if print_results:
        print(f"\n {'-' * 20}", flush=True)
        print_performance(actual, pred , overall_acc)
        print("Recall_std ", np.array(rec_w).std())
        print("f1_std ", np.array(f1_w).std())
    return actual, pred, overall_acc

def run_1DCNN_experiment(label, overlap, sequence_length, random_state, batch_size, folds = 10, print_results = True):
    device = torch.device('cpu')
    print(f"For [{label}]: Stride {overlap}% || Length of {sequence_length} notes, {batch_size}", flush=True)
    df, pitch_distr_sections, ql_distr_sections = prepare_dataframe(label, overlap, sequence_length)
    prefixes, prefixes_y, prefix_to_ids = return_prefixes(df, label)
    df_windowed = normalize_and_traspose(df, label, pitch_distr_sections, ql_distr_sections)
    input_size = 6
    num_classes = len(LABEL_LIST_TRAIN[label])
    learning_rate = 1e-3

    num_epochs = 200
    patience = 10
    # EMBEDDING Parameters
    num_keys = len(df['key'].unique())
    key_embed_dim = len(df['key'].unique())
    # GLOBAL FEATURES Parameters
    global_feat_dim = 64
    global_out_dim = 128

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    f1_w=[]
    rec_w=[]
    actual = []
    pred = []
    overall_acc = []
    plt_val_loss = []
    plt_train_loss = []
    print(f"Processing fold:", end="" , flush=True)
    for fold in range(folds):
        patience = 10
        print(f"Fold {fold}:", flush=True)
        #train_prefixes, test_prefixes = train_test_split(prefixes, stratify=prefixes_y, test_size=0.2, random_state=random_state)
        train_prefixes_temp, test_prefixes, train_prefixes_y_temp, test_prefixes_y = train_test_split(prefixes, prefixes_y, stratify=prefixes_y, test_size=0.1, random_state=random_state * fold)
        train_prefixes, val_prefixes, train_prefixes_y, val_prefixes_y = train_test_split(train_prefixes_temp, train_prefixes_y_temp, stratify=train_prefixes_y_temp, test_size=0.11, random_state=random_state * fold)


        # Step 3: Gather IDs for train and test sets
        train_ids = [id_ for prefix in train_prefixes for id_ in prefix_to_ids[prefix]]
        test_ids = [id_ for prefix in test_prefixes for id_ in prefix_to_ids[prefix]]
        val_ids = [id_ for prefix in val_prefixes for id_ in prefix_to_ids[prefix]]

        X_train, y_train = df_windowed[df_windowed['section_id'].isin(train_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(train_ids)][label]
        X_test, y_test = df_windowed[df_windowed['section_id'].isin(test_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(test_ids)][label]
        X_val, y_val = df_windowed[df_windowed['section_id'].isin(val_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(val_ids)][label]

        # 1. Get sequences as lists
        all_sequences_train = X_train['NoteAndRest'].tolist()
        all_sequences_val = X_val['NoteAndRest'].tolist()
        all_sequences_test = X_test['NoteAndRest'].tolist()

        # 2. Flatten training sequences and split off last column
        flat_train = np.concatenate(all_sequences_train, axis=0)
        scaler = StandardScaler().fit(flat_train[:, :-1])  # Fit on all but last column

        # 3. Transform each sequence (all but last col), and reattach the last column
        def normalize_sequence(seq):
            seq = np.array(seq)
            norm_part = scaler.transform(seq[:, :-1])
            return np.hstack([norm_part, seq[:, -1:]])

        # 4. Apply to all splits
        X_train['NoteAndRest'] = [normalize_sequence(seq) for seq in all_sequences_train]
        X_val['NoteAndRest']   = [normalize_sequence(seq) for seq in all_sequences_val]
        X_test['NoteAndRest']  = [normalize_sequence(seq) for seq in all_sequences_test]

        # Unique to dl

        global_train = np.vstack(X_train['global_features'].to_numpy())
        global_test = np.vstack(X_test['global_features'].to_numpy())
        global_val = np.vstack(X_val['global_features'].to_numpy())
        # Fit scaler on all columns except last
        scaler = StandardScaler().fit(global_train[:, :-1])

        # Transform and reattach last column
        X_train['global_features'] = list(np.hstack([scaler.transform(global_train[:, :-1]), global_train[:, -1:]]))
        X_test['global_features'] = list(np.hstack([scaler.transform(global_test[:, :-1]), global_test[:, -1:]]))
        X_val['global_features'] = list(np.hstack([scaler.transform(global_val[:, :-1]), global_val[:, -1:]]))



        # We prepare the train, test and validation loaders
        dataset_train = NoteSequenceDataset(X_train['NoteAndRest'].tolist(), X_train['global_features'].tolist(), y_train.tolist())
        train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        dataset_test = NoteSequenceDataset(X_test['NoteAndRest'].tolist(), X_test['global_features'].tolist(), y_test.tolist())
        test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
        dataset_val = NoteSequenceDataset(X_val['NoteAndRest'].tolist(), X_val['global_features'].tolist(), y_val.tolist())
        val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

        model = CNN(input_size, num_classes,
                        num_keys, key_embed_dim, global_feat_dim, global_out_dim).to(device)

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train.tolist()), y=y_train.tolist())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight = class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = StepLR(optimizer= optimizer, step_size=20, gamma=0.1)
        """scheduler = CosineAnnealingWarmRestarts(
            optimizer= optimizer,
            T_0 = 20,
            T_mult = 2,
            eta_min = 1e-6
        )"""
        best_loss = float('inf')

        best_model = model.state_dict()
        for epoch in range(num_epochs):
            model.train()

            for batch_idx, (sequences, global_feats, labels) in enumerate(train_loader):
                # Reset gradients
                optimizer.zero_grad()   # Float keys are converted into long
                output = model(sequences, global_feats[:, -1].long(), global_feats[:, :-1])  # Assuming global_feats[:, 0] is the key, global_feats[:, 1:] is the other features
                loss = criterion(output, labels)
                loss.backward()


                """
                # Just for visualization
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                if batch_idx % 10 == 0:
                    print("Targets:", labels.tolist(), flush=True)
                    print("Preds:", preds.tolist(), flush=True)
                    print("Probs:", probs.tolist(), flush=True)
                    print("Train loss: ", loss, flush=True)
                """

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                # Update
                optimizer.step()

            plt_train_loss.append(loss.item())
            # VALIDATION Step
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, global_feats, labels in val_loader:
                    output = model(sequences, global_feats[:, -1].long(), global_feats[:, :-1])
                    loss = criterion(output, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            #print("Val loss", avg_val_loss, flush=True)
            #print("Epoch", epoch, flush=True)
            plt_val_loss.append(avg_val_loss)
            if avg_val_loss > best_loss:
                patience -= 1
            else:
                best_loss = avg_val_loss
                patience = 10
                best_model = model.state_dict()

            if patience <= 0:
                break
            val_acc = correct / total
            #print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}, Best Val Loss = {best_loss:.4f}, Val Acc = {val_acc:.4f}, Patience = {patience:.4f},", flush=True)
            scheduler.step()

        print(f"Finished at epoch {epoch}", flush=True)
        model.load_state_dict(best_model)
        y_pred, y_actual, y_accuracy = (check_accuracy(test_loader, model))
        pred.extend(y_pred)
        actual.extend(y_actual)
        overall_acc.append(y_accuracy)
        rec_w.append(recall_score(y_actual, y_pred, average='weighted'))
        f1_w.append(f1_score(y_actual, y_pred, average='weighted'))
    if print_results:
        print(f"\n {'-' * 20}", flush=True)
        print_performance(actual, pred , overall_acc)
        print("Recall_std ", np.array(rec_w).std())
        print("f1_std ", np.array(f1_w).std())
        print_performance(y_actual, y_pred , y_accuracy)
    return actual, pred, overall_acc

def run_1DCNN_experiment_t(label, overlap, sequence_length, random_state, batch_size, folds = 10, print_results = True):
    device = torch.device('cpu')
    print(f"For [{label}]: Stride {overlap}% || Length of {sequence_length} notes, {batch_size}", flush=True)
    df, pitch_distr_sections, ql_distr_sections = prepare_dataframe(label, overlap, sequence_length)
    prefixes, prefixes_y, prefix_to_ids = return_prefixes(df, label)
    df_windowed = normalize_and_traspose(df, label, pitch_distr_sections, ql_distr_sections)
    input_size = 6
    num_classes = len(LABEL_LIST_TRAIN[label])
    learning_rate = 1e-3

    num_epochs = 200
    patience = 10
    # EMBEDDING Parameters
    num_keys = len(df['key'].unique())
    key_embed_dim = len(df['key'].unique())
    # GLOBAL FEATURES Parameters
    global_feat_dim = 64
    global_out_dim = 128

    rec_w =[]
    f1_w = []
    actual = []
    pred = []
    overall_acc = []
    
    
    print(f"Processing fold:", end="" , flush=True)
    for fold in range(folds):
        patience = 10
        print(f"Fold {fold}:", flush=True)
        #train_prefixes, test_prefixes = train_test_split(prefixes, stratify=prefixes_y, test_size=0.2, random_state=random_state)

        train_prefixes, test_prefixes, val_prefixes, _, __, ___= train_test_val_split(prefixes, prefixes_y, random_state * fold)


        # Step 3: Gather IDs for train and test sets
        train_ids = [id_ for prefix in train_prefixes for id_ in prefix_to_ids[prefix]]
        test_ids = [id_ for prefix in test_prefixes for id_ in prefix_to_ids[prefix]]
        val_ids = [id_ for prefix in val_prefixes for id_ in prefix_to_ids[prefix]]

        X_train, y_train = df_windowed[df_windowed['section_id'].isin(train_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(train_ids)][label]
        X_test, y_test = df_windowed[df_windowed['section_id'].isin(test_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(test_ids)][label]
        X_val, y_val = df_windowed[df_windowed['section_id'].isin(val_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(val_ids)][label]

        # 1. Get sequences as lists
        all_sequences_train = X_train['NoteAndRest'].tolist()
        all_sequences_val = X_val['NoteAndRest'].tolist()
        all_sequences_test = X_test['NoteAndRest'].tolist()

        # 2. Flatten training sequences and split off last column
        flat_train = np.concatenate(all_sequences_train, axis=0)
        scaler = StandardScaler().fit(flat_train[:, :-1])  # Fit on all but last column

        # 3. Transform each sequence (all but last col), and reattach the last column
        def normalize_sequence(seq):
            seq = np.array(seq)
            norm_part = scaler.transform(seq[:, :-1])
            return np.hstack([norm_part, seq[:, -1:]])

        # 4. Apply to all splits
        X_train['NoteAndRest'] = [normalize_sequence(seq) for seq in all_sequences_train]
        X_val['NoteAndRest']   = [normalize_sequence(seq) for seq in all_sequences_val]
        X_test['NoteAndRest']  = [normalize_sequence(seq) for seq in all_sequences_test]

        # Unique to dl

        global_train = np.vstack(X_train['global_features'].to_numpy())
        global_test = np.vstack(X_test['global_features'].to_numpy())
        global_val = np.vstack(X_val['global_features'].to_numpy())
        # Fit scaler on all columns except last
        scaler = StandardScaler().fit(global_train[:, :-1])

        # Transform and reattach last column
        X_train['global_features'] = list(np.hstack([scaler.transform(global_train[:, :-1]), global_train[:, -1:]]))
        X_test['global_features'] = list(np.hstack([scaler.transform(global_test[:, :-1]), global_test[:, -1:]]))
        X_val['global_features'] = list(np.hstack([scaler.transform(global_val[:, :-1]), global_val[:, -1:]]))



        # We prepare the train, test and validation loaders
        dataset_train = NoteSequenceDataset(X_train['NoteAndRest'].tolist(), X_train['global_features'].tolist(), y_train.tolist())
        train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        dataset_test = NoteSequenceDataset(X_test['NoteAndRest'].tolist(), X_test['global_features'].tolist(), y_test.tolist())
        test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
        dataset_val = NoteSequenceDataset(X_val['NoteAndRest'].tolist(), X_val['global_features'].tolist(), y_val.tolist())
        val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

        model = CNN(input_size, num_classes,
                        num_keys, key_embed_dim, global_feat_dim, global_out_dim).to(device)

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train.tolist()), y=y_train.tolist())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight = class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = StepLR(optimizer= optimizer, step_size=20, gamma=0.1)
        """scheduler = CosineAnnealingWarmRestarts(
            optimizer= optimizer,
            T_0 = 20,
            T_mult = 2,
            eta_min = 1e-6
        )"""
        best_loss = float('inf')

        best_model = model.state_dict()
        for epoch in range(num_epochs):
            model.train()

            for batch_idx, (sequences, global_feats, labels) in enumerate(train_loader):
                # Reset gradients
                optimizer.zero_grad()   # Float keys are converted into long
                output = model(sequences, global_feats[:, -1].long(), global_feats[:, :-1])  # Assuming global_feats[:, 0] is the key, global_feats[:, 1:] is the other features
                loss = criterion(output, labels)
                loss.backward()


                """
                # Just for visualization
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                if batch_idx % 10 == 0:
                    print("Targets:", labels.tolist(), flush=True)
                    print("Preds:", preds.tolist(), flush=True)
                    print("Probs:", probs.tolist(), flush=True)
                    print("Train loss: ", loss, flush=True)
                """

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                # Update
                optimizer.step()

            
            # VALIDATION Step
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, global_feats, labels in val_loader:
                    output = model(sequences, global_feats[:, -1].long(), global_feats[:, :-1])
                    loss = criterion(output, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            #print("Val loss", avg_val_loss, flush=True)
            #print("Epoch", epoch, flush=True)
            
            if avg_val_loss > best_loss:
                patience -= 1
            else:
                best_loss = avg_val_loss
                patience = 10
                best_model = model.state_dict()

            if patience <= 0:
                break
            val_acc = correct / total
            #print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}, Best Val Loss = {best_loss:.4f}, Val Acc = {val_acc:.4f}, Patience = {patience:.4f},", flush=True)
            scheduler.step()

        print(f"Finished at epoch {epoch}", flush=True)
        model.load_state_dict(best_model)
        y_pred, y_actual, y_accuracy = (check_accuracy(test_loader, model))
        pred.extend(y_pred)
        actual.extend(y_actual)
        overall_acc.append(y_accuracy)
        rec_w.append(recall_score(y_actual, y_pred, average='weighted'))
        f1_w.append(f1_score(y_actual, y_pred, average='weighted'))
    if print_results:
        print(f"\n {'-' * 20}", flush=True)
        print_performance(actual, pred , overall_acc)
        print("Recall_std ", np.array(rec_w).std())
        print("f1_std ", np.array(f1_w).std())
        print_performance(y_actual, y_pred , y_accuracy)
    return actual, pred, overall_acc

def run_GNN_experiment(label, subgraph_size, num_hidden_features, random_state,num_layers, dropout = 0.7, batch_size = 16, folds = 10):
    
    
    num_epochs = 100
    num_input_features = 9
    global_out_dim = 64
    num_output_features = len(LABEL_LIST_TRAIN[label])
    patience = 10
    metadata = (
        ['note'],
        [('note', 'onset', 'note'),
        ('note', 'consecutive', 'note'),
        ('note', 'rest', 'note'),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ('note', 'consecutive', 'note'),
        ('note', 'rest', 'note'),
        ('note', 'consecutive_rev', 'note'),
        ('note', 'rest_rev', 'note')]
        )
    print(f"Running for subgraph_size = {subgraph_size}, dropout = {dropout}, num_hidden_features = {num_hidden_features}, num_layers = {num_layers}, batch_size {batch_size}", flush=True)
    graphs, graphs_y = load_graphs_perLabel(label)

    for i in range(folds):
        print(f"Loaded {len(graphs)} graphs",  flush=True)
        X_train, X_test, X_val, y_train, y_test, y_val= train_test_val_split(graphs, graphs_y, random_state * i)


        class GraphDataset(Dataset):
            def __init__(self, graphs, labels):
                self.graphs = graphs
                self.labels = labels

            def __len__(self):
                return len(self.graphs)

            def __getitem__(self, idx):
                data = self.graphs[idx]
                data.y = self.labels[idx]
                return data

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataset = GraphDataset(X_train, y_train)
        test_dataset = GraphDataset(X_test, y_test)
        val_dataset = GraphDataset(X_val, y_val)

        train_loader = MuseNeighborLoader(train_dataset, subgraph_size=subgraph_size, batch_size=batch_size,
                                        num_neighbors=[int(subgraph_size * 0.6), int(subgraph_size * 0.3), int(subgraph_size * 0.1)])
        test_loader = MuseNeighborLoader(test_dataset, subgraph_size=subgraph_size, batch_size=batch_size,
                                        num_neighbors=[int(subgraph_size * 0.6), int(subgraph_size * 0.3), int(subgraph_size * 0.1)])
        val_loader = MuseNeighborLoader(val_dataset, subgraph_size=subgraph_size, batch_size=batch_size,
                                        num_neighbors=[int(subgraph_size * 0.6), int(subgraph_size * 0.3), int(subgraph_size * 0.1)])

        print(len(train_loader), flush=True)
        print(len(test_loader), flush=True)
        print(len(val_loader), flush=True)
        # Placeholders for global features
        train_globals = []
        val_globals = []
        test_globals = []

        def attach_global_features(data):
            steps = [midi_to_note(int(x.item())) for x in data['note'].x[:, 6]]
            pitch_distr = list(compute_avg_folded_hist_labeled_notes(steps, data['note'].x[:, 3]))
            ql_distr = get_folded_rhythm_histogram(data['note'].x[:, 3])
            global_feat = extract_feature_graph(data['note'].x, data['note'].primitive_global_features[0][0], data['note'].primitive_global_features[0][1], pitch_distr, ql_distr)  # your function
            return np.array(list(global_feat.values()), dtype=np.float32)


        # Extract global features from each dataset (example logic)
        for data in train_dataset:
            global_feat = attach_global_features(data)
            train_globals.append(global_feat)
            data['note'].derived_global_features = global_feat

        for data in val_dataset:
            global_feat = attach_global_features(data)
            val_globals.append(global_feat)
            data['note'].derived_global_features = global_feat
            #data.global_features = global_feat

        for data in test_dataset:
            global_feat = attach_global_features(data)
            test_globals.append(global_feat)
            data['note'].derived_global_features = global_feat
            #data.global_features = global_feat


        # Stack into arrays for normalization
        train_globals_array = np.stack(train_globals)
        val_globals_array = np.stack(val_globals)
        test_globals_array = np.stack(test_globals)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_globals_array)
        val_scaled = scaler.transform(val_globals_array)
        test_scaled = scaler.transform(test_globals_array)
        # Collect all note features from the training set
        all_note_features = torch.cat([data['note'].x for data in train_dataset], dim=0)
        scaler_notes = StandardScaler()
        scaler_notes.fit(all_note_features.numpy())


        def reassign_scaled_values(dataset, scaled):
            for data, scaled_feat in zip(dataset, scaled):
                data['note'].derived_global_features = torch.tensor(scaled_feat, dtype=torch.float32).unsqueeze(0)
                data['note'].x = torch.tensor(
                    scaler_notes.transform(data['note'].x.numpy()), dtype=torch.float32
                )
        reassign_scaled_values(train_dataset, train_scaled)
        reassign_scaled_values(test_dataset, test_scaled)
        reassign_scaled_values(val_dataset, val_scaled)

        df_notes = pd.read_json('note_corpus3.json', orient ='split', compression = 'infer')
        num_keys = len(df_notes['key'].unique())
        key_embed_dim = len(df_notes['key'].unique())

        model = MetricalGNN(num_input_features, num_hidden_features, num_output_features, num_layers, num_keys, key_embed_dim, metadata, 63, global_out_dim, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = StepLR(optimizer= optimizer, step_size=20, gamma=0.1)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        best_loss = float('inf')
        train_loss = None


        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
            model.train()
            total_train_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                batch = to_device_batch(batch, device)
                x_dict = {'note': batch['note'].x}
                edge_index_dict = extract_edge_index_dict(batch)

                optimizer.zero_grad()

                key_tensor = batch['note'].primitive_global_features[:, 0].long().to(device)
                derived_features = batch['note'].derived_global_features[:, :-1].to(device)

                out = model(x_dict, edge_index_dict, batch['note'].batch.to(device), key_tensor, derived_features)
                loss = criterion(out, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                total_train_loss += loss.item()
                #print(f"  Batch {batch_idx}: loss={loss.item():.4f}", end='\r', flush=True)

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, print_results = False)
            #print(f"\nTrain Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", flush=True)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                early_stop_counter = 0
                #print("  ✅ New best model saved.", flush=True)
            else:
                early_stop_counter += 1
                #print(f"  ⏳ No improvement. Patience left: {patience - early_stop_counter}", flush=True)

            if early_stop_counter >= patience:
                #print("  🛑 Early stopping triggered.", flush=True)
                break

        # Load best model for final evaluation or saving
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), 'best_metrical_gnn.pt')


        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, print_results=False)
        return test_accuracy
    #print(f"Avg. loss of {test_loss:2f} || Accuracy of {test_accuracy:2f}", flush=True)

def run_RNN_experiment(label, overlap, sequence_length, num_layers, hidden_size, random_state, batch_size = 16, folds = 10, print_results = True):

    actual = []
    pred = []
    overall_acc = []
    rec_w = []
    f1_w = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"For [{label}]: Stride {overlap}% || Length of {sequence_length} notes, num layer {num_layers}, hidden_size {hidden_size}, batch size {batch_size}", flush=True)

    df, pitch_distr_sections, ql_distr_sections = prepare_dataframe(label, overlap, sequence_length)
    prefixes, prefixes_y, prefix_to_ids = return_prefixes(df, label)
    df_windowed = normalize_and_traspose(df, label, pitch_distr_sections, ql_distr_sections)

    input_size = 6
    num_classes = len(LABEL_LIST_TRAIN[label])
    learning_rate = 1e-3

    num_epochs = 200
    patience = 10

    num_keys = len(df['key'].unique())
    key_embed_dim = 4
    global_feat_dim = 64
    global_out_dim = 128

    for i in range(10):
        random_state = 42 * i
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



        train_prefixes_temp, test_prefixes, train_prefixes_y_temp, test_prefixes_y = train_test_split(
            prefixes, prefixes_y, stratify=prefixes_y, test_size=0.1, random_state=random_state
        )
        train_prefixes, val_prefixes, train_prefixes_y, val_prefixes_y = train_test_split(
            train_prefixes_temp, train_prefixes_y_temp, stratify=train_prefixes_y_temp, test_size=0.11, random_state=random_state
        )

        train_ids = [id_ for prefix in train_prefixes for id_ in prefix_to_ids[prefix]]
        test_ids = [id_ for prefix in test_prefixes for id_ in prefix_to_ids[prefix]]
        val_ids = [id_ for prefix in val_prefixes for id_ in prefix_to_ids[prefix]]

        X_train, y_train = df_windowed[df_windowed['section_id'].isin(train_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(train_ids)][label]
        X_test, y_test = df_windowed[df_windowed['section_id'].isin(test_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(test_ids)][label]
        X_val, y_val = df_windowed[df_windowed['section_id'].isin(val_ids)][['NoteAndRest', 'global_features']], df_windowed[df_windowed['section_id'].isin(val_ids)][label]

        # 1. Get sequences as lists
        all_sequences_train = X_train['NoteAndRest'].tolist()
        all_sequences_val = X_val['NoteAndRest'].tolist()
        all_sequences_test = X_test['NoteAndRest'].tolist()

        # 2. Flatten training sequences and split off last column
        flat_train = np.concatenate(all_sequences_train, axis=0)
        scaler = StandardScaler().fit(flat_train[:, :-1])  # Fit on all but last column

        # 3. Transform each sequence (all but last col), and reattach the last column
        def normalize_sequence(seq):
            seq = np.array(seq)
            norm_part = scaler.transform(seq[:, :-1])
            return np.hstack([norm_part, seq[:, -1:]])

        # 4. Apply to all splits
        X_train['NoteAndRest'] = [normalize_sequence(seq) for seq in all_sequences_train]
        X_val['NoteAndRest']   = [normalize_sequence(seq) for seq in all_sequences_val]
        X_test['NoteAndRest']  = [normalize_sequence(seq) for seq in all_sequences_test]

        # Unique to dl

        global_train = np.vstack(X_train['global_features'].to_numpy())
        global_test = np.vstack(X_test['global_features'].to_numpy())
        global_val = np.vstack(X_val['global_features'].to_numpy())
        # Fit scaler on all columns except last
        scaler = StandardScaler().fit(global_train[:, :-1])

        # Transform and reattach last column
        X_train['global_features'] = list(np.hstack([scaler.transform(global_train[:, :-1]), global_train[:, -1:]]))
        X_test['global_features'] = list(np.hstack([scaler.transform(global_test[:, :-1]), global_test[:, -1:]]))
        X_val['global_features'] = list(np.hstack([scaler.transform(global_val[:, :-1]), global_val[:, -1:]]))

        dataset_train = NoteSequenceDataset(X_train['NoteAndRest'].tolist(), X_train['global_features'].tolist(), y_train.tolist())
        train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
        dataset_test = NoteSequenceDataset(X_test['NoteAndRest'].tolist(), X_test['global_features'].tolist(), y_test.tolist())
        test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
        dataset_val = NoteSequenceDataset(X_val['NoteAndRest'].tolist(), X_val['global_features'].tolist(), y_val.tolist())
        val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

        model = RNN(input_size, hidden_size, num_layers, num_classes,
                    num_keys, key_embed_dim, global_feat_dim, global_out_dim).to(device)

        best_model = model.state_dict()
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train.tolist()), y=y_train.tolist())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.1)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            for sequences, global_feats, labels in train_loader:
                sequences = sequences.to(device)
                global_feats = global_feats.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = model(sequences, global_feats[:, -1].long(), global_feats[:, :-1])
                loss = criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()

            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, global_feats, labels in val_loader:
                    sequences = sequences.to(device)
                    global_feats = global_feats.to(device)
                    labels = labels.to(device)

                    output = model(sequences, global_feats[:, -1].long(), global_feats[:, :-1])
                    loss = criterion(output, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(test_loader)
            if avg_val_loss > best_loss:
                patience -= 1
            else:
                best_loss = avg_val_loss
                patience = 10
                best_model = model.state_dict()

            if patience <= 0:
                break

            val_acc = correct / total
            #print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}, Best Val Loss = {best_loss:.4f}, Val Acc = {val_acc:.4f}, Patience = {patience}", flush=True)
            scheduler.step()

        print(f"Finished at epoch {epoch}", flush=True)
        model.load_state_dict(best_model)

        def check_accuracy(loader, model):
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            all_accuracy = []
            with torch.no_grad():
                for sequences, global_feats, labels in loader:
                    sequences = sequences.to(device)
                    global_feats = global_feats.to(device)
                    labels = labels.to(device)

                    output = model(sequences, global_feats[:, -1].long(), global_feats[:, :-1])
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_accuracy.append(correct / total)
            accuracy = correct / total
            return all_preds, all_labels, all_accuracy

        y_pred, y_actual, y_accuracy = check_accuracy(test_loader, model)
        actual.extend(y_actual)
        pred.extend(y_pred)
        overall_acc.extend(y_accuracy)
        rec_w.append(recall_score(y_actual, y_pred, average='weighted'))
        f1_w.append(f1_score(y_actual, y_pred, average='weighted'))
    print_performance(actual, pred, overall_acc)
    print("Recall_std ", np.array(rec_w).std())
    print("f1_std ", np.array(f1_w).std())
    return y_pred, y_actual, y_accuracy
