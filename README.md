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
2. [Project Structure](#structure)  
3. [Citations](#citations)
4. [Corpus Analysis](overview_data.ipynb)
5. [Replicate Experiments](run_experiments.ipynb)


---

## Overview

Building upon [[1]](#1), our results show that
data-driven Machine Learning approaches provide a significant improve-
ment for the aforementioned classification tasks compared to model-
based Artificial Intelligence. Moreover, we introduce a method based
on a Graph Convolutional Neural Network (GNN) architecture that ex-
ploits the relationships between music components. To the best of our
knowledge, this is the first application of a GNN to Non-Western MIR.
This work has the potential to set a new baseline for state-of-the-art
methods which identify nawba and ṭāb‘.

---

###  Institution:
 Faculty of Engineering, Free University of Bolzano-Bozen  


##  Contributors

- **Alessandro Sellani** 
- **Ivan Donadello** 
- **Niccolo' Pretto** 

---

## Notebooks Overview
There are two main Jupyter notebooks. In [overview_data](overview_data.ipynb) the user can explore the Arab-Andalusian corpus by statistical analysis and class-balance investigation along with the (nawba, tab) relationship.
While [run_experiments](run_experiments.ipynb) allows the user to replicate the experiments to achieve the results listed in the paper

```
project-root/
│
├── corpus-dataset/                # Core corpus track-wise and meta data
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
├── overview_data.ipynb            # Data analysis and corpus statistics
├── run_experiments.ipynb          # Experiments for all models
│
├── environment.yml                # Environment dependencies
```

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
---
<a id="1">[1]</a> 
Pretto, N., Bozkurt, B., Caro Repetto, R., Serra, X.: Nawba recognition for
Arab-Andalusian music using templates from music scores. In: Proceedings of the
15th Sound and Music Computing Conference, SMC 2018. p. 394 – 399. Limassol,
Cyprus (2018). https://doi.org/10.5281/zenodo.1257388



