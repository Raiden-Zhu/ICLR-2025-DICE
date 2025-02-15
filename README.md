# ICLR-2025-DICE

This project conducts three kinds of experiments and provides a reproducible experimental environment. The following is a detailed guide on how to install dependencies and run the experiments.

## Table of Contents
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation
### Prerequisites
- Ensure you have Python installed on your system.
- `pip` should be available for package installation.

### Steps
1. Clone the repository to your local machine:
```bash
git clone https://github.com/Raiden-Zhu/ICLR-2025-DICE.git
cd ICLR-2025-DICE
```
2. Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Running Experiments
This project contains three sets of experiments, each corresponding to a different directory. You can run the experiments by following these steps:

### Experiment 1
The purpose of this experiment is to calculate the actual and estimated values of the data influence under specific experimental conditions. The calculation results will be written into the corresponding JSON files in the `experiment1/loss_record` folder.
Navigate to the `experiment1` directory and execute the `run.sh` script:
```bash
cd experiment1
bash run_repeat.sh
```
The results of Experiment 1 will be generated after the script finishes execution.

### Experiment 2
The purpose of this experiment is to calculate the influence indicators of normal nodes and abnormal nodes. The calculation results will be written into the corresponding JSON files in the `experiment2/loss_record` folder.
Similarly, for Experiment 2, move to the `experiment2` directory and run the `run.sh` script:
```bash
cd ../experiment2
bash run_repeat.sh
```
The corresponding results will be available once the script completes.

### Experiment 3
The purpose of this experiment is to calculate the influence values of specific nodes on multi - level neighbors. The calculation results will be written into the corresponding JSON files in the `experiment3/loss_record` folder.
To run Experiment 3, enter the `experiment3` directory and execute the `run.sh` script:
```bash
cd ../experiment3
bash run_repeat.sh
```
This will generate the results for Experiment 3.

## Project Structure
```plaintext
ICLR-2025-DICE/
│
├── experiment1/
    └──loss_record
    └──networks
    └──our_datasets
    └──utils
    └──workers
    └──main_new.py
    └── run_repeat.sh
├── experiment2/
├── experiment3/
├── requirements.txt
└── README.md
```

### Citation

Please cite our paper if you find this repo useful in your work:

```
@inproceedings{
zhu2025dice,
title={{DICE}: Data Influence Cascade in Decentralized Learning},
author={Tongtian Zhu and Wenhao Li and Can Wang and Fengxiang He},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=2TIYkqieKw}
}
```
