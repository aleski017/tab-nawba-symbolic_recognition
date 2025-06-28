#  Melodic Pattern Analysis for Arab-Andalusian Music

The majority of Music Information Retrieval (MIR) research
is Western-centric, and the limited availability of annotated resources
poses a challenge for data-intensive approaches. In this work, we imple-
ment data-driven models and analyse their classification performance in
two fundamental concepts in Arab-Andalusian music: nawba and ṭāb‘
using symbolic encoding. To address data scarcity, we employ two data
augmentation strategies: sliding window segmentation and graph sub-
sampling. We process a dataset of Arab-Andalusian digital scores to
extract meaningful symbolic features and provide the resulting dataset
for experiment reproduction and further research.

---

##  Table of Contents



1. [Libraries Installation](#installation)  
2. [Notebooks Overview](#usage)  
4. [Replicating Experiments](#replicating-experiments)  
5. [Citations](#citations)

---

## Overview

Our results show that
data-driven Machine Learning approaches provide a significant improve-
ment for the aforementioned classification tasks compared to model-
based Artificial Intelligence. Moreover, we introduce a method based
on a Graph Convolutional Neural Network (GNN) architecture that ex-
ploits the relationships between music components. To the best of our
knowledge, this is the first application of a GNN to Non-Western MIR.
This work has the potential to set a new baseline for state-of-the-art
methods which identify nawba and ṭāb‘

---

###  Institution:
 Faculty of Engineering, Free University of Bolzano-Bozen  


##  Contributors

- **Alessandro Sellani** 
- **Ivan Donadello** 
- **Niccolo' Pretto** 

---

##  Installation

```bash
# Clone the repository
git https://github.com/aleski017/tab-nawba-symbolic_recognition.git
cd tab-nawba-symbolic_recognition

# Create a virtual environment 
conda env create -f environment.yml

# Activate the environment
conda activate pyg-env
```

## Notebooks Overview

```
project-root/
│
├── corpus-dataset/                # Core corpus data and metadata
│   ├── documents/                 # Track-wise data
│   ├── andalusian_description.json
│   ├── andalusian_form.json
│   ├── andalusian_mizan.json
│   ├── andalusian_nawba.json
│   ├── andalusian_recording.json
│   └── andalusian_tab.json
│
├── experiments/                   
│   ├── run_experiments.py         # Main Experiment Scripts
│   ├── graphs/                    # Graph-converted XML Scores
│   └── utilities/                 # Helper Functions
│       ├── constants.py
│       ├── corpus_search.py       
│       ├── dl_utilities.py        # Deep Learning Architectures
│       ├── features_eng.py
│       ├── model_matching.py
│       └── temporal_analysis.py
│
├── note_corpus3.json
│
├── graph_findings.ipynb           # Notebook for graph exploration
├── overview_data.ipynb            # Data overview and inspection
├── run_experiments.ipynb          # Interactive version of experiment runner
├── section_matching.ipynb         # Notebook for section-level analysis
│
├── environment.yml                # Conda environment dependencies
└── README.md
```

