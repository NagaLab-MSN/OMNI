# OMNI — Chemical–Gene Interaction Predictor

> A Graph Neural Network framework for classifying interaction types between chemicals and genes across a heterogeneous biological knowledge graph.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-red?style=flat-square)](https://www.pytorchlightning.ai/)
[![DGL](https://img.shields.io/badge/DGL-Deep%20Graph%20Library-orange?style=flat-square)](https://www.dgl.ai/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

**OMNI** predicts how a given chemical interacts with a gene — not just *whether* an interaction exists, but *what kind*. Given a chemical–gene pair, the model outputs ranked probabilities across a comprehensive set of interaction categories (e.g., `increases^expression`, `decreases^activity`).

The model learns from a rich heterogeneous biological graph that integrates chemicals, genes, diseases, and pathways, capturing both local neighborhood structure and long-range global dependencies.

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-Relation Classification** | Predicts specific interaction type, not just binary link presence |
| **Heterogeneous Graph** | Integrates chemicals, genes, diseases, and pathways in a single unified graph |
| **Hybrid Attention Encoder** | Combines local GAT-based attention with global RWR-based long-range context |
| **Reproducible Pipeline** | Built on PyTorch Lightning for clean training, checkpointing, and evaluation |

---

## Model Architecture

The model consists of three stages:

### 1. Node Embedding Layer
Assigns a unique, trainable feature vector to every node in the graph.

### 2. GNN Encoder — `HeteroRelGAT`
The core of the model. Produces context-rich embeddings by combining two complementary representations:

- **Local Representation** — Relation-specific Graph Attention (GAT) layers aggregate features from direct neighbors, with separate attention weights per relationship type.
- **Global Representation** — Random Walk with Restart (RWR) identifies influential long-range nodes; a global attention mechanism incorporates these into each node's embedding.

### 3. Edge Decoder
A prediction head that takes a chemical embedding and a gene embedding and applies a dedicated MLP for each possible interaction type, outputting a ranked score distribution.

---

## Getting Started

### Prerequisites

- Python 3.8+
- `conda` or `venv` for environment management
- A CUDA-enabled GPU is strongly recommended for training

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/NagaLab-MSN/OMNI.git
cd OMNI
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

**3. Install PyTorch** (select the build matching your CUDA version):
```
https://pytorch.org/get-started/locally/
```

**4. Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Prepare Your Data

Place all required CSV files (see [Data Format](#data-format) below) into a single directory. On the first run, the training script will automatically preprocess the files, build the graph, and cache artifacts for fast subsequent loads.

### Step 2 — Train a Model

```bash
python main_train.py --base_data_path /path/to/your/data_directory
```

This will:
- Preprocess raw data and build the heterogeneous graph (first run only)
- Run training and validation loops
- Save two checkpoints:
  - `best-model-multi-rel.ckpt` — highest validation AUROC
  - `final_model_multi_rel.ckpt` — final epoch weights

To view all configurable hyperparameters:
```bash
python main_train.py --help
```

### Step 3 — Predict for a Chemical–Gene Pair

```bash
python main_predict_manual.py \
    --chemical_id  "D000041" \
    --gene_id      "1017" \
    --model_checkpoint  final_model_multi_rel.ckpt \
    --base_data_path    /path/to/your/data_directory
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--chemical_id` | ✅ | Unique identifier of the query chemical |
| `--gene_id` | ✅ | Unique identifier of the query gene |
| `--model_checkpoint` | ✅ | Path to the trained `.ckpt` file |
| `--base_data_path` | ✅ | Path to the data directory (needed to load graph structure and node IDs) |

---

## Data Format

All files must be CSV and placed in the directory specified by `--base_data_path`. The column names below are required exactly as listed.

| File | Required Columns |
|---|---|
| `CTD_chem_gene_ixns.csv` | `ChemicalID`, `GeneID`, `InteractionActions` |
| `chemical_chemical_noNaN.csv` | `Chemical1_name1`, `Chemical2_name2` |
| `CTD_chemicals_diseases.csv` | `ChemicalID`, `DiseaseID` |
| `CTD_chem_pathways_enriched.csv` | `ChemicalID`, `PathwayID` |
| `CTD_genes_diseases.csv` | `GeneID`, `DiseaseID` |
| `CTD_genes_pathways.csv` | `GeneID`, `PathwayID` |
| `gene_gene.csv` | `Gene 1`, `Gene 2` |

---

## Results

The `results/` folder contains pre-computed outputs:

- **`Table_S1`** — Targets and drugs from the pan-cancer proteogenomics study, with predicted interaction scores.
- **Gene-specific folders** — Each contains a specific gene paired with a set of chemicals and their predicted interaction probabilities.
- **VDR** — Prediction probabilities calculated across nearly all chemicals in the dataset.

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---


## Acknowledgements

This work builds on the following open-source libraries:

- [PyTorch](https://pytorch.org/) — Core deep learning framework
- [Deep Graph Library (DGL)](https://www.dgl.ai/) — Graph neural network building blocks
- [PyTorch Lightning](https://www.pytorchlightning.ai/) — Training infrastructure and checkpointing