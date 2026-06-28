# MA-Net for Combine Harvester Time-Series Prediction

This repository provides a compact PyTorch implementation of the Memory-Attention Network (MA-Net) used for multivariate time-series prediction in combine harvester operation. The code is intended to support reproducibility of the manuscript by providing the core model modules, configuration file, training entry point, and a small anonymized/synthetic-format sample dataset.

> Note: The complete field dataset is not included because of confidentiality restrictions related to agricultural machinery operation and field trials.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── configs/
│   └── default.yaml
├── data_sample/
│   ├── sample_ch_data.csv
│   └── data_description.md
├── main.py
└── manet/
    ├── __init__.py
    ├── data.py
    ├── model.py
    ├── train.py
    └── utils.py
```

## Model modules

The implementation contains the main MA-Net components described in the manuscript:

- `HistoricalKnowledgeVectorDatabase`: HKVD with cosine-similarity retrieval and threshold-based memory update.
- `LocalFeatureExtractionUnit`: LFEU based on residual convolution, batch normalization, LeakyReLU activation, and pooling.
- `GlobalDependencyModelingUnit`: GDMU based on a lightweight attention module and feed-forward network.
- `HybridFeatureFusionUnit`: HFFU for fusing the current global representation with retrieved long-term memory.
- `BidirectionalMemoryRecallUnit`: BMRU based on a bidirectional LSTM decoder.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick smoke test

Run a random forward-pass check without field data:

```bash
python main.py smoke --config configs/default.yaml
```

Expected output:

```text
input shape:  (4, 64, 6)
output shape: (4, 40, 6)
```

## Demo training with sample data

```bash
python main.py train --config configs/default.yaml --data data_sample/sample_ch_data.csv
```

The sample CSV is provided only to demonstrate the expected data format. It should not be used to reproduce the field results reported in the manuscript.

## Input data format

The default configuration expects a CSV file containing the following columns:

```text
HV,TCS,FS,BR,IR,LR
```

where `HV`, `TCS`, and `FS` are operating parameters, and `BR`, `IR`, and `LR` are harvesting-quality-related performance metrics.

## Configuration

The default configuration in `configs/default.yaml` uses:

- input length: 64 time steps
- prediction length: 40 time steps
- input dimension: 6
- output dimension: 6
- chronological train/validation split to reduce temporal leakage risk in the public demo

## Data availability

The complete raw field dataset is not publicly released because of confidentiality agreements related to field experiments and agricultural machinery operation. A small anonymized/synthetic-format sample file is provided to illustrate the input format and support basic code execution.

## Citation

If you use this code, please cite the associated manuscript. The `CITATION.cff` file can be updated after publication with the final article DOI.
