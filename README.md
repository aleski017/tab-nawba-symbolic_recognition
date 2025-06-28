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



1. [Packages installation](#installation)  
2. [Notebooks Overview](#usage)  
3. [Project Structure](#project-structure) 
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

# Create a virtual environment (optional but recommended)
conda env create --name ENV-NAME --file environment.yml

