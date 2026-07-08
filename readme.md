# Deep Learning Assignment

This repository contains implementations, notebooks, datasets, and analysis reports for the Deep Learning assignment.
Most experiments are performed in **Google Colab notebooks**, while reusable components are implemented as **Python modules (`.py` files)** and imported into the notebooks.

---

# Project Structure

ai24btech11018_dl_Assignment_1/
│
├── Question1/
│   ├── question1_dl.ipynb
│   └── dl_1_analysis.pdf
│
├── Question2/
│   └── dl_2_analysis.pdf
│
├── Question3/
│   └── dl_3_analysis.pdf
│
├── Question4/
│   ├── data_generation.py
│   ├── iit_h_mess_dataset_5000.csv
│   └── dl_4_analysis.pdf
│
├── Question5/
│   ├── adaline.py
│   ├── adaline_exp.ipynb
│   └── dl_5_analysis.pdf
│
├── Question6_6_7/
│   ├── mlp.py
│   ├── activations.py
│   ├── weights.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── 6_7_experiments.ipynb
|   |__adaline.py
│   └── dl_6_analysis.pdf
│   |__DL_7_ANALYSIS.pdf
|
└── readme.md




**Files**
## Question 1 
* `question1_dl.ipynb` – Implementation and experiments
* `dl_1_analysis.pdf` – Analysis and observations

## Question 2

**File**

* `dl.pdf` – Theory and analysis


## Question 3

**File**

* `dl1.pdf` – Analysis and discussion

---

## Question 4 – Dataset Generation

**Files**

* `data_generation.py` – Script used to generate the dataset
* `iit_h_mess_dataset_5000.csv` – Generated dataset with 5000 samples
* `dl_4_analysis.pdf` – Dataset generation explanation

The generated dataset is used in later questions.

---

## Question 5 – ADALINE Implementation

**Files**

* `adaline.py` – Implementation of the ADALINE algorithm
* `adaline_exp.ipynb` – Notebook where `adaline.py` is imported and used
* `dl_5_analysis.pdf` – Analysis of ADALINE experiments

**Dataset used:**
`iit_h_mess_dataset_5000.csv` generated in Question 4.

Section **5.1 analysis** is included in `dl_5_analysis.pdf`.

---
## Module and Notebook Usage

### MLP Implementation

The file **`M_lp.py`** contains the implementation of the Multi-Layer Perceptron (MLP).
Inside `M_lp.py`, the following modules are imported:

* `activations.py` – contains activation functions used in the network
* `weights.py` – contains weight initialization methods
* `losses.py` – contains loss functions used during training
* `optimizers.py` – contains optimization algorithms for updating weights

These modules are used internally by `M_lp.py`.

---

### Notebook Experiments

All experiments for **Question 6 and Question 7** are performed in the notebook:

`6_7_experiments.ipynb`

In this notebook:

* `M_lp.py` is imported to train and evaluate the MLP model.
* The dataset **`iit_h_mess_dataset_5000.csv`** is loaded for training and testing.

---

### Question 7 Additional Model

For Question 7, the notebook also imports:

* `adaline.py`

Thus, Question 7 experiments use both:

* `M_lp.py` (MLP model)
* `adaline.py` (ADALINE model)

The dataset **`iit_h_mess_dataset_5000.csv`** generated earlier is used for all experiments.


# Dataset

The dataset used in the assignment was generated using:

`data_generation.py`

Output dataset:

`iit_h_mess_dataset_5000.csv`

This dataset is used in **Questions 4, 5, 6, and 7**.

---
Note on Theory and Analysis

The theoretical explanations and detailed analysis for each question are provided in the corresponding PDF analysis files.

dl_1_analysis.pdf – Theory and analysis for Question 1

dl_2_analysis.pdf – Theory and analysis for Question 2

dl_3_analysis.pdf – Theory and analysis for Question 3

dl_4_analysis.pdf – Dataset generation explanation for Question 4

dl_5_analysis.pdf – ADALINE analysis (includes Section 5.1)

dl_6_analysis.pdf – MLP experiments and analysis (includes Section 6.1)

dl_7_analysis.pdf – Kernel methods, neural feature extraction, and kernel comparison


# Tools Used

* Python
* NumPy
* Pandas
* Matplotlib
* Google Colab / Jupyter Notebook
