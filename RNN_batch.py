print("Results for 1DCNN", flush=True)
for label in ["tab", "nawba"]:
    for overlap in [0, 0.1, 0.3 ,0.5]:
        for sequence_length in [128, 256, 512]:
        
            for num_layers in [2, 3, 4]:
                for batch_size in [32, 64, 128]:
                    run_1DCNN_experiment(label, overlap, sequence_length, 42, num_layers, 0.7, batch_size)

from experiments.run_experiments import *
print("Results for RNN", flush=True)
for label in ["tab", "nawba"]:
    for overlap in [0, 0.1, 0.3 ,0.5]:
        for sequence_length in [128, 256, 512]:
            for hidden_size in [64, 128, 256]:
                for num_layers in [2, 3, 4]:
                    for batch_size in [16, 32,64]:
                        run_RNN_experiment(label, overlap, sequence_length, num_layers, hidden_size,42, batch_size=batch_size)
