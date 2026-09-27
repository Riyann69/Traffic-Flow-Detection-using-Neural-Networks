# Traffic Flow Detection Using Neural Networks

Five-minute-ahead congestion classification (Low / Medium / High) for individual freeway detectors in the
**PeMS08** dataset. This project compares an **ANN**, a **1-D CNN** and a hybrid **Quantum Neural Network (QNN)**
against simple baselines, evaluated on a strictly chronological, leakage-free split.

📄 Paper: `Traffic_Flow_Detection_Using_Neural_Networks.pdf`

---

## Task

- **Data:** PeMS08, 170 loop detectors in San Bernardino, California. Each detector reports flow, occupancy and speed every 5 minutes, 1 Jul – 31 Aug 2016.
- **Input:** a 2-hour window (24 steps × 3 features) from one detector.
- **Target:** that detector's flow class at the next 5-minute step. Low, Medium and High are split at the detector's own 33rd and 66th percentiles from the training period.
- **Split:** chronological. The first 80% of the timeline is used for training; its last 10% is held out for validation. The final 20% is the test set, 606,390 windows.
- **Scaling:** thresholds and per-detector z-score scaling are computed from the training period only.

---



## Results (test set)


| Model                       | Accuracy   | Macro-F1   | Mean IoU   | Parameters |
| --------------------------- | ---------- | ---------- | ---------- | ---------- |
| **1-D CNN**                 | **89.18%** | **0.8911** | **0.8073** | 51,235     |
| ANN                         | 88.99%     | 0.8893     | 0.8044     | 19,779     |
| 15-min average (rule)       | 88.08%     | 0.8804     | 0.7904     | –          |
| Persistence (rule)          | 87.28%     | 0.8721     | –          | –          |
| Logistic regression         | 87.24%     | 0.8717     | –          | 219        |
| QNN (6 qubits)              | 86.25%     | 0.8625     | 0.7633     | 2,681      |
| Classical control (6 units) | 86.13%     | 0.8610     | 0.7613     | –          |
| Linear trend (rule)         | 80.66%     | 0.8043     | –          | –          |
| Majority class              | 33.63%     | 0.1678     | –          | –          |


**Key findings**

- Traffic five minutes ahead is highly predictable from current conditions: a simple 15-minute average rule already reaches 88%.
- The CNN and ANN beat the best rule by about 1 percentage point (McNemar p < 10⁻¹⁵⁰). They do so on 121 and 108 of the 170 detectors, respectively.
- The 6-qubit QNN performs on par with a size-matched classical layer (86.25% vs 86.13%). The quantum circuit shows no measurable advantage in this setting.

---



## How to run

Run the notebooks in this order. Each one saves its outputs for the next.


| #   | Notebook                  | What it does                                                                        | Time (CPU) |
| --- | ------------------------- | ----------------------------------------------------------------------------------- | ---------- |
| 1   | `preprocess.ipynb`        | Builds windows, labels, split and scaling → `data/traffic_flow_preprocessed.npz`    | ~1 min     |
| 2   | `baselines.ipynb`         | Rule-based baselines and logistic regression                                        | ~2 min     |
| 3   | `train_ann.ipynb`         | Multi-layer perceptron (72 → 128 → 64 → 32 → 3)                                     | ~5 min     |
| 4   | `train_cnn.ipynb`         | 1-D CNN over the 24-step window                                                     | ~15 min    |
| 5   | `train_qnn.ipynb`         | PCA (6 components) → 6-qubit PennyLane circuit → dense head, plus classical control | ~80 min    |
| 6   | `Models_comparison.ipynb` | Comparison tables, charts, per-detector analysis, McNemar tests                     | ~1 min     |


Before running, set `PROJECT` at the top of each notebook to your own folder path.

---



## Folder layout

```
├── preprocess.ipynb
├── baselines.ipynb
├── train_ann.ipynb
├── train_cnn.ipynb
├── train_qnn.ipynb
├── Models_comparison.ipynb
├── data/
│   └── pems08.npz                    # raw dataset
├── models/                           # saved models
├── results/                          # *_results.json, *_predictions.npz, baseline_results.json
├── requirements.txt
└── README.md
```

---



## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.11, TensorFlow 2.21 (Keras 3) and PennyLane 0.45.

---



## Model details

- **ANN:** the 24 × 3 window is flattened to 72 inputs. Dense layers of 128, 64 and 32 ReLU units, with dropout 0.2 after the first, feed a 3-way softmax.
- **1-D CNN:**
  - two Conv1D layers with 32 filters, then max pooling;
  - a Conv1D with 64 filters, then max pooling;
  - a Conv1D with 128 filters, then global average pooling;
  - a dense layer of 128 units with dropout 0.3, then a 3-way softmax.
- **QNN:**
  - PCA compresses the 72 inputs to 6 components (92.8% of the variance), scaled to [0, π];
  - each component is angle-encoded as an RX rotation on one of 6 qubits;
  - 3 strongly entangling layers follow (54 trainable angles), and the ⟨Z⟩ expectation values are read out;
  - a dense head of 64 and 32 units with dropout 0.3 feeds a 3-way softmax;
  - simulation uses PennyLane's `default.qubit` with backpropagation;
  - training uses a 50,000-window subset, but evaluation covers the full test set.
- **Classical control:** identical to the QNN, except the circuit is replaced by a Dense(6, tanh) layer.

All networks use Adam, sparse categorical cross-entropy, early stopping on validation loss (patience 3) and seed 42.

---



## Dataset

PeMS08 was collected by the Caltrans Performance Measurement System (PeMS) and released with ASTGCN
(Guo et al., AAAI 2019).

---



## Author

Riyan Wankhede, VIT-AP University