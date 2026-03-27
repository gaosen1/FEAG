# FEAG: Fine-grained Explicit Alignment and Gating Framework for Multimodal Sentiment Analysis

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-orange" alt="Coming Soon"/>
</p>

> **Code coming soon.** This repository will host the official PyTorch implementation of the FEAG framework.  
> Please ⭐ star the repo and watch for updates!

---

## Overview

**FEAG** is a multimodal sentiment analysis (MSA) framework designed to handle the challenges of noisy, unaligned multimodal streams (text, audio, and vision). It combines three core technical contributions:

1. **Explicit Alignment Supervision (EAS)** — Uses word-level timestamps to construct a pseudo-ground-truth alignment map, then constrains cross-attention distributions via a KL-divergence auxiliary loss, ensuring that attention heads focus on semantically corresponding cross-modal regions rather than learning spurious correlations.

2. **Bidirectional Query-Dependent Gating (BQDG)** — A lightweight gating mechanism applied to both directions of cross-modal information flow (text→audio, text→vision and audio/vision→text). The gate is conditioned on the query representation so that uninformative or noisy signals are dynamically suppressed before they influence the fusion output.

3. **Cross-Modal Transformer (CMT)** — A standard transformer cross-attention block that serves as the backbone for inter-modal interaction. EAS and BQDG are integrated into this block to impose structural supervision and dynamic filtering simultaneously.

FEAG achieves competitive results on CMU-MOSI, CMU-MOSEI, and CH-SIMS by combining explicit alignment constraints with adaptive noise filtering, without requiring manual forced-alignment annotations at inference time.

---

## Architecture

```
          ┌──────────────────────────────────────────────────────────┐
          │                        FEAG Model                        │
          │                                                          │
          │  Text Encoder      Audio Encoder     Vision Encoder      │
          │  (BERT / RoBERTa)  (Transformer)     (Transformer)       │
          │        │                 │                  │            │
          │        ▼                 ▼                  ▼            │
          │  ┌──────────────────────────────────────────────────┐    │
          │  │           Cross-Modal Transformer (CMT)          │    │
          │  │                                                  │    │
          │  │   ┌─────────────────────────────────────────┐   │    │
          │  │   │   Explicit Alignment Supervision (EAS)  │   │    │
          │  │   │   • Pseudo GT map from timestamps        │   │    │
          │  │   │   • KL-divergence alignment loss         │   │    │
          │  │   └─────────────────────────────────────────┘   │    │
          │  │                                                  │    │
          │  │   ┌─────────────────────────────────────────┐   │    │
          │  │   │  Bidirectional Query-Dependent Gating   │   │    │
          │  │   │  (BQDG)                                 │   │    │
          │  │   │   • Query-conditioned gate vector        │   │    │
          │  │   │   • Bidirectional noise suppression      │   │    │
          │  │   └─────────────────────────────────────────┘   │    │
          │  └──────────────────────────────────────────────────┘    │
          │                         │                                │
          │                         ▼                                │
          │                 Fusion & Prediction                      │
          │               (Sentiment Score / Label)                  │
          └──────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role |
|-----------|------|
| **Cross-Modal Transformer (CMT)** | Bidirectional cross-attention backbone connecting all three modalities |
| **Explicit Alignment Supervision (EAS)** | Converts word-level timestamps into a soft alignment prior; penalizes attention that deviates from this prior via KL divergence |
| **Bidirectional Query-Dependent Gating (BQDG)** | Learns a per-token gate conditioned on the query to filter noisy cross-modal signals before fusion |
| **Fusion Head** | Aggregates gated multimodal representations and predicts continuous sentiment scores or discrete labels |

---

## Supported Datasets

| Dataset | Language | Modalities | Labels | Split |
|---------|----------|------------|--------|-------|
| [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) | English | Text + Audio + Vision | Continuous \[-3, 3\] | Train / Val / Test |
| [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) | English | Text + Audio + Vision | Continuous \[-3, 3\] | Train / Val / Test |
| [CH-SIMS](https://github.com/thuiar/MMSA) | Chinese | Text + Audio + Vision | Continuous \[-1, 1\] | Train / Val / Test |

All datasets are used in their **unaligned** (raw) forms. Word-level timestamps are loaded to construct the alignment prior used by EAS; no forced-alignment preprocessing is required at inference time.

---

## Planned Project Structure

```
FEAG/
├── data/
│   ├── dataset.py          # MultimodalDataset — loads unaligned sequences & timestamps
│   └── collate.py          # Padding & batching utilities
├── models/
│   ├── feag.py             # Top-level FEAG model
│   ├── cross_modal_transformer.py   # CMT block
│   ├── alignment.py        # Explicit Alignment Supervision (EAS)
│   └── gating.py           # Bidirectional Query-Dependent Gating (BQDG)
├── utils/
│   ├── metrics.py          # Acc-2/5/7, F1, MAE, Pearson r
│   └── logger.py           # Training logger
├── train.py                # Training & evaluation entry point
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Evaluation Metrics

FEAG will be evaluated using the standard MSA benchmark metrics:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error between predicted and ground-truth sentiment scores |
| **Pearson r** | Pearson correlation coefficient |
| **Acc-7** | 7-class accuracy (sentiment score rounded to nearest integer in \[-3, 3\]) |
| **Acc-5** | 5-class accuracy (for CH-SIMS \[-1, 1\] scale) |
| **Acc-2** | Binary (positive / negative) accuracy |
| **F1** | Weighted F1-score for the binary classification |

---

## Training Objective

The total training loss combines a primary regression loss with an auxiliary alignment loss:

```
L_total = L_task + λ · L_align
```

where:
- **L_task** — MSE or L1 regression loss between the predicted sentiment score and the ground-truth label.
- **L_align** — KL-divergence loss between the cross-attention distribution and the pseudo-GT alignment map produced by EAS.
- **λ** — A scalar hyperparameter balancing the two objectives (typically 0.1 – 1.0).

---

## Requirements *(planned)*

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.23.0
scikit-learn>=1.2.0
tqdm>=4.65.0
```

---

## Getting Started *(coming soon)*

```bash
# 1. Clone the repository
git clone https://github.com/gaosen1/FEAG.git
cd FEAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare your dataset
#    Download CMU-MOSI / CMU-MOSEI / CH-SIMS and update the data path in config.

# 4. Train
python train.py --dataset mosi --lr 1e-4 --epochs 40 --lambda_align 0.5
```

---

## Citation *(coming soon)*

If you find this work useful, please consider citing:

```bibtex
@article{feag2026,
  title   = {Fine-grained Explicit Alignment and Gating Framework for Multimodal Sentiment Analysis},
  author  = {},
  journal = {},
  year    = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

We gratefully acknowledge the publicly available datasets CMU-MOSI, CMU-MOSEI, and CH-SIMS, as well as the open-source toolkits [MMSA](https://github.com/thuiar/MMSA) and [Hugging Face Transformers](https://github.com/huggingface/transformers) that this work builds upon.
