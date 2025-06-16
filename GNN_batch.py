
print("Results for GNN", flush=True)
for label in ["tab", "nawba"]:
    for subgraph_size in [128, 256, 512]:
        for num_hidden_features in [64, 128, 256]:
            for num_layers in [2, 3, 4]:
                for batch_size in [32, 64, 128]:
                    run_GNN_experiment(label, subgraph_size, num_hidden_features, 42, num_layers, 0.7, batch_size)

for overlap in [0, 0.1, 0.3 ,0.5]:
    run_KNN_experiment("tab", overlap, 1024, 42)

for label in ["tab", "nawba"]:
    for overlap in [0, 0.1, 0.3 ,0.5]:
        for sequence_length in [256, 512, 1024]:
            run_KNN_experiment(label, overlap, sequence_length, 42)

from experiments.run_experiments import *
print("Results for KNN", flush=True)
for overlap in [0, 0.1, 0.3 ,0.5]:
    run_KNN_experiment("tab", overlap, 1024, 42)

for label in ["tab", "nawba"]:
    for overlap in [0, 0.1, 0.3 ,0.5]:
        for sequence_length in [256, 512, 1024]:
            run_KNN_experiment(label, overlap, sequence_length, 42)

print("Results for SVC", flush=True)
for label in ["tab", "nawba"]:
    for overlap in [0, 0.1, 0.3 ,0.5]:
        for sequence_length in [128, 256, 512]:
            run_LinearSVC_experiment(label, overlap, sequence_length, 42)