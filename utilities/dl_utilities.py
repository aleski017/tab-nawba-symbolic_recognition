import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, AttentionalAggregation
from torch_geometric.nn import SAGEConv, HeteroConv
from graphmuse.nn.conv.gat import CustomGATConv
from graphmuse.utils.graph_utils import trim_to_layer
from torch_geometric.utils import trim_to_layer as trim_to_layer_pyg
from utilities.model_matching_computation import print_performance
from utilities.constants import DF_PATH_TRACKS
import partitura
import numpy as np
import os
from utilities.corpus_search import *
from sklearn.preprocessing import LabelEncoder
import graphmuse as gm
from scipy import stats
from collections import Counter

class NoteSequenceDataset(Dataset):
    def __init__(self, sequences, global_features, labels):
        self.sequences = sequences
        self.global_features = global_features
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Returns the lists into a converted torch for RNN usage
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        global_feat = torch.tensor(self.global_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sequence, global_feat, label

def check_accuracy(loader, model, print_predictions = False):
    model.eval()
    total = 0
    correct = 0
    pred = []
    actual = []
    with torch.no_grad():
        for sequences, global_feats, labels in loader:
            #data, targets = batch
            output = model(sequences, global_feats[:, -1].long(), global_feats[:, :-1])
            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if print_predictions:
                print(f"Predictions: {predicted.tolist()}")
                print(f"Actual: {labels.tolist()}")
            pred.extend(predicted.tolist())
            actual.extend(labels.tolist())

    accuracy = float(correct) / float(total) * 100
    print(f'Got {correct} / {total} correct with accuracy {accuracy:.2f}', flush=True)
    model.train()
    return actual, pred, accuracy

class CNN(nn.Module):
    def __init__(self, input_size, num_classes, num_keys, key_embed_dim, global_feat_dim, global_out_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.AdaptiveMaxPool1d(1)

        )

        self.key_embedding = nn.Embedding(num_keys, key_embed_dim)

        self.global_fc = nn.Sequential(
            nn.Linear(global_feat_dim, global_out_dim),

            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + key_embed_dim + global_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x, key, global_feats):

        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        key_vec = self.key_embedding(key)
        global_vec = self.global_fc(global_feats)

        combined = torch.cat([x, key_vec, global_vec], dim=1)
        out = self.classifier(combined)
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                    num_keys, key_embed_dim, global_feat_dim, global_out_dim, dropout= 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.key_embedding = nn.Embedding(num_keys, key_embed_dim)
        self.global_fc = nn.Sequential(
            nn.Linear(global_feat_dim, global_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + key_embed_dim + global_out_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),              # added intermediate layer

            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, key, global_feats):
        lstm_out, (h_n, _) = self.lstm(x)
        backward = h_n[-1]
        key_vec = self.key_embedding(key)
        global_vec = self.global_fc(global_feats)
        combined = torch.cat([backward, key_vec, global_vec], dim=1)
        out = self.classifier(combined)
        return out

################################## GRAPH UTILITIES ##############################################################################################################################################
# Create a GNN Encoder
class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, edge_types, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(
            HeteroConv(
                {
                    edge_type: SAGEConv(input_channels, hidden_channels, normalize=True, project=True)
                    for edge_type in edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers-1):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(hidden_channels, hidden_channels)
                    for edge_type in edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node,
                neighbor_mask_edge):

        for i, conv in enumerate(self.convs[:-1]):
            if not neighbor_mask_edge is None and not neighbor_mask_node is None:
                x_dict, edge_index_dict, _ = trim_to_layer(
                    layer=self.num_layers - i,
                    neighbor_mask_node=neighbor_mask_node,
                    neighbor_mask_edge=neighbor_mask_edge,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        if not neighbor_mask_edge is None and not neighbor_mask_node is None:
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=1,
                neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge,
                x=x_dict,
                edge_index=edge_index_dict,
            )
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        return x_dict


class HierarchicalHeteroGraphConv(torch.nn.Module):
    def __init__(self, edge_types, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(
            HeteroConv(
                {
                    edge_type: CustomGATConv(input_channels, hidden_channels, heads=4, add_self_loops=False)
                    for edge_type in edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers-1):
            conv = HeteroConv(
                {
                    edge_type: CustomGATConv(hidden_channels, hidden_channels, heads=4, add_self_loops=False)
                    for edge_type in edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node,
                neighbor_mask_edge):

        for i, conv in enumerate(self.convs[:-1]):
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=self.num_layers - i,
                neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge,
                x=x_dict,
                edge_index=edge_index_dict,
            )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Last layer
        x_dict, edge_index_dict, _ = trim_to_layer(
            layer=1,
            neighbor_mask_node=neighbor_mask_node,
            neighbor_mask_edge=neighbor_mask_edge,
            x=x_dict,
            edge_index=edge_index_dict,
        )
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        return x_dict


class FastHierarchicalHeteroGraphConv(torch.nn.Module):
    def __init__(self, edge_types, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(
            HeteroConv(
                {
                    edge_type: SAGEConv(input_channels, hidden_channels, normalize=True, project=True)
                    for edge_type in edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers - 1):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(hidden_channels, hidden_channels)
                    for edge_type in edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None,
                neighbor_mask_edge=None):

        for i, conv in enumerate(self.convs):
            if not neighbor_mask_edge is None and not neighbor_mask_node is None:
                x_dict, edge_index_dict, _ = trim_to_layer_pyg(
                    layer=i,
                    num_sampled_edges_per_hop=neighbor_mask_edge,
                    num_sampled_nodes_per_hop=neighbor_mask_node,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs) - 1:
                x_dict = {key: x.relu() for key, x in x_dict.items()}
                x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        return x_dict


class GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(GRUWrapper, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class MetricalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_keys, key_embed_dim, metadata, global_feat_dim, global_out_dim, dropout, fast=False):
        super(MetricalGNN, self).__init__()
        self.num_layers = num_layers

        if fast:
            self.gnn = FastHierarchicalHeteroGraphConv(metadata[1], input_dim, hidden_dim, num_layers)
            self.fhs = True
        else:
            self.gnn = HierarchicalHeteroGraphSage(metadata[1], input_dim, hidden_dim, num_layers)
            self.fhs = False

        # === Add GlobalAttention pooling here ===
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Optional, improves interpretability of attention weights
        )
        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            nn=nn.Identity()  # or any transformation on node features before aggregation
        )

        self.global_fc = nn.Sequential(
            nn.Linear(global_feat_dim, global_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.key_embedding = nn.Embedding(num_keys, key_embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_out_dim + key_embed_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),              # added intermediate layer

            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )


    def forward(self, x_dict, edge_index_dict, batch, key, global_feats, neighbor_mask_node=None, neighbor_mask_edge=None):
        x_dict = self.gnn(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        note = x_dict["note"]

        key_vec = self.key_embedding(key)
        global_vec = self.global_fc(global_feats)
        pooled = self.att_pool(note, batch)

        combined = torch.cat([pooled, global_vec, key_vec], dim=1)
        out = self.mlp(combined)
        return out

def to_device_batch(batch, device):
    # Moves all batch tensors to the specified device
    for node_type in batch.node_types:
        batch[node_type].x = batch[node_type].x.to(device)
        batch[node_type].batch = batch[node_type].batch.to(device)
    for edge_type in batch.edge_types:
        batch[edge_type].edge_index = batch[edge_type].edge_index.to(device)
    batch.y = batch.y.to(device)
    return batch

def extract_edge_index_dict(batch):
    return {edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}

def evaluate(model, loader, criterion, device, print_results=True):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    actual = []
    prediction = []
    accuracy_list = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device_batch(batch, device)
            x_dict = {'note': batch['note'].x}
            edge_index_dict = extract_edge_index_dict(batch)

            key_tensor = batch['note'].primitive_global_features[:, 0].long().to(device)
            derived_features = batch['note'].derived_global_features[:, :-1].to(device)
            batch_tensor = batch['note'].batch.to(device)
            target = batch.y.to(device)

# Forward pass
            out = model(x_dict, edge_index_dict, batch_tensor, key_tensor, derived_features)

            loss = criterion(out, target)







            total_loss += loss.item()
            pred = out.argmax(dim=1)
            prediction.extend(pred.cpu().numpy())
            actual.extend(batch.y.cpu().numpy())

            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            accuracy_list.append(correct / total)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    if print_results:
        print(f"\n {'-' * 50}")
        print_performance(actual, prediction, accuracy_list)
    return avg_loss, accuracy

def save_graphs():

    le = LabelEncoder()
    df_notes = pd.read_json('note_corpus3.json', orient ='split', compression = 'infer')
    keys_to_encode = df_notes['key'].apply(lambda x: str(x)).tolist()
    df_notes['key'] =  le.fit_transform(keys_to_encode)
    all_note_features = []
    all_global_features = []
    all_raw_features = []
    tracks_ids = []
    for track in get_all_id_tracks():

        path = DF_PATH_TRACKS + track + '/' + track + '.xml'

        if not os.path.exists(path): continue


        score = partitura.load_musicxml(DF_PATH_TRACKS + track + '/' + track + '.xml')
        tracks_ids.append(track)

        note_array = score.note_array(include_pitch_spelling=True, include_key_signature=True, include_metrical_position=True)
        all_raw_features.append(note_array)
        # Convert note_array into feature matrix
        feature_list = []

        bpm = df_notes[df_notes["section_id"].str.contains(track, na=False)]['bpm'].mean()
        key = df_notes[df_notes["section_id"].str.contains(track, na=False)]['key'].mode()[0]

        letter_notes = []
        for note in note_array:

            onset_beat = note['onset_beat']
            duration_beat = note['duration_beat']
            onset_quarter = note['onset_quarter'] #to maybe delete
            duration_quarter = note['duration_quarter'] #to maybe delete
            onset_div = note['onset_div'] #to maybe delete
            duration_div = note['duration_div'] #to maybe delete
            pitch = note['pitch']
            voice = note['voice'] #to maybe delete
            alter_string =  '-'*abs(note['alter'])if note['alter'] <= -1 else str('#'*abs(note['alter']))
            step = note['step'] + alter_string
            letter_notes.append(step)
            #print(step)
            is_downbeat = note['is_downbeat']



            feature_list.append([onset_beat,duration_beat,onset_quarter,duration_quarter,onset_div,duration_div,pitch,voice,is_downbeat])
        feature_list = np.array(feature_list)
        #pitch_distr = list(compute_avg_folded_hist_labeled_notes(letter_notes, feature_list[:, 3]))
        #ql_distr = get_folded_rhythm_histogram(feature_list[:, 3])
        all_note_features.append(feature_list)
        #all_global_features.append(extract_feature_graph(feature_list, key, bpm, pitch_distr, ql_distr))
        all_global_features.append([key, bpm])



    global_feature_array = np.array(all_global_features, dtype=np.float32)
    global_feature_array = torch.tensor(global_feature_array, dtype=torch.float32)

    for note_feat, global_feat, raw_feat, track_id in zip(all_note_features, global_feature_array, all_raw_features, tracks_ids):
        score_graph = gm.create_score_graph(note_feat, raw_feat)
        #score_graph['note'][f'x_std'] = scaler.fit_transform(feature_list[:, :10])
        for l in ["tab", "nawba", "mizan"]:
            l_value = df_notes[df_notes["section_id"].str.contains(track_id, na=False)][l].iloc[0]
            score_graph['note'][f'y_{l}'] = torch.tensor([l_value], dtype=torch.long)
            score_graph['note']['primitive_global_features'] = global_feat.unsqueeze(0)

        torch.save(score_graph, f'graphs/{track_id}.pt')

def extract_feature_graph(sequence, key, bpm, pitch_distr, ql_distr=None):
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

    onsets = [note[0] for note in sequence]
    pitches = [note[6] for note in sequence]
    durations = [note[1] for note in sequence]



    beatStrengths = [note[8] for note in sequence]
    length_section = len(pitches)
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) -1)]
    num_ascending_intervals = len([interval for interval in intervals if interval > 0])

    key_value = key
    bpm = bpm
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
    "mean_offset": np.mean(onsets),
    "mean_strengths": np.mean(beatStrengths),
    "mean_duration": np.mean(durations),
    "duration_changes": sum_duration_changes,
    #"duration_range": np.max(durations) - np.min(durations),
    #"dominant_melodic_interval" : dominant_interval,
    "num_ascending_intervals" : num_ascending_intervals,
    "number_distinct_melodic_intervals" : number_distinct_melodic_intervals,
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







