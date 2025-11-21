# Chemical-Gene Interaction Predictor (GNN)

This repository provides a complete framework for training and using a Graph Neural Network (GNN) to predict specific interaction types between chemicals and genes. The model is designed to work with a complex, heterogeneous biological graph, integrating various entities like diseases and pathways to learn powerful, context-aware node representations.

Built on **PyTorch**, **Deep Graph Library (DGL)**, and **PyTorch Lightning**, this project is designed for reproducibility, modularity, and high performance.

## Key Features

-   **Multi-Relation Classification**: Instead of simple link prediction, this model classifies a chemical-gene interaction into one of many specific categories (e.g., `increases^expression`, `decreases^activity`).
-   **Heterogeneous Graph Model**: Seamlessly integrates different biological entities (chemicals, genes, diseases, pathways) and the relationships between them.
-   **Hybrid Attention Encoder**: The core GNN employs a sophisticated hybrid attention strategy to learn node embeddings:
    -   **Local Neighborhood Attention**: Uses relation-specific Graph Attention (GAT) layers to aggregate information from immediate neighbors.
    -   **Global Graph Attention**: Leverages Random Walk with Restart (RWR) to identify and attend to important, long-range nodes, capturing the global context of each entity.
-   **Clean & Modular Codebase**: The project is organized into distinct modules for data handling, model architecture, and execution, making it easy to understand, maintain, and extend.
-   **Efficient Data Pipeline**: Features a one-time data preprocessing and caching mechanism. Raw CSV files are converted into a graph format and saved to disk for instant loading in subsequent runs.

## Model Architecture

The model is comprised of three primary stages:

1.  **Node Embedding Layer**: Assigns a unique, trainable feature vector (embedding) to every node in the graph.
2.  **GNN Encoder (`HeteroRelGAT`)**: This is the core of the model. It generates a final, context-rich embedding for each node by combining two distinct representations:
    -   A **local representation** is learned by aggregating features from direct neighbors, with separate attention weights for each type of relationship.
    -   A **global representation** is learned by attending over a wider, more influential neighborhood identified by RWR, allowing the model to incorporate long-range dependencies.
3.  **Edge Decoder**: A prediction head that takes the final embeddings of a chemical and a gene. It contains a separate MLP for each possible interaction type, which outputs a score indicating the likelihood of that specific interaction.


## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

-   Python 3.8+
-   `conda` or `venv` for environment management
-   A CUDA-enabled GPU is highly recommended for training.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/chemical-gene-gnn.git
    cd chemical-gene-gnn
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install PyTorch:**
    Install a version of PyTorch compatible with your CUDA toolkit. Follow the official instructions here: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

4.  **Install All Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

The project workflow is split into two main parts: training a model and then using that model for prediction.

### 1. Data Setup

You must provide your own data in CSV format. Place all the required source files (listed in the **Data Format** section below) into a single directory. The first time you run the training script, it will automatically process these files, build the graph, and create cached artifacts for fast re-loading.

### 2. Training a New Model

To train the GNN, run the `main_train.py` script. The only mandatory argument is the path to your data directory.

```bash
python main_train.py --base_data_path /path/to/your/data_directory
```

-   This command will preprocess the data (if needed), execute the training and validation loops, and save two model checkpoints:
    -   `best-model-multi-rel.ckpt`: The checkpoint with the best validation AUROC score.
    -   `final_model_multi_rel.ckpt`: The checkpoint from the very last training epoch.
-   To see all configurable hyperparameters (e.g., embedding dimensions, learning rate, epochs), run `python main_train.py --help`.

### 3. Predicting Interactions for a Single Pair

After training, you can use the `main_predict_manual.py` script to get a ranked list of all possible interaction types for any given chemical-gene pair.

```bash
python main_predict_manual.py \
    --chemical_id "D000041" \
    --gene_id "1017" \
    --model_checkpoint final_model_multi_rel.ckpt \
    --base_data_path /path/to/your/data_directory
```

**Required Arguments:**

| Argument             | Description                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| `--chemical_id`      | The unique ID of the chemical you want to query.                                                    |
| `--gene_id`          | The unique ID of the gene you want to query.                                                        |
| `--model_checkpoint` | Path to the trained `.ckpt` model file you want to use for inference.                               |
| `--base_data_path`   | Path to the original data directory. This is essential for loading the graph structure and node IDs. |


## Data Format

The data processing pipeline expects the following CSV files to be present in the directory specified by `--base_data_path`. The column names listed here are required.

| File Name                        | Required Columns                             |
| -------------------------------- | -------------------------------------------- |
| `CTD_chem_gene_ixns.csv`         | `ChemicalID`, `GeneID`, `InteractionActions` |
| `chemical_chemical_noNaN.csv`    | `Chemical1_name1`, `Chemical2_name2`         |
| `CTD_chemicals_diseases.csv`     | `ChemicalID`, `DiseaseID`                    |
| `CTD_chem_pathways_enriched.csv` | `ChemicalID`, `PathwayID`                    |
| `CTD_genes_diseases.csv`         | `GeneID`, `DiseaseID`                        |
| `CTD_genes_pathways.csv`         | `GeneID`, `PathwayID`                        |
| `gene_gene.csv`                  | `Gene 1`, `Gene 2`                           |

## Results
The result folder contains the calculatated results with the Code

Table_S1 Results for the targets and drugs obtained from pan-cancer proteogenomics study
All other folder contains a specific gene with a specific chemicals with interaction probablities
Note: In case of VDR, the prediction probablities were calculated for almost all the chemicals


## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

This implementation is made possible by the excellent work of the teams behind these open-source libraries:
-   [PyTorch](https://pytorch.org/)
-   [Deep Graph Library (DGL)](https://www.dgl.ai/)
-   [PyTorch Lightning](https://www.pytorchlightning.ai/)


