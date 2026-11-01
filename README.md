# Human Activity Recognition (HAR) Using Discrete Hidden Markov Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline implemented from scratch in Python to classify sequential human activities (**Still**, **Standing**, **Walking**, **Jumping**) from mobile tri-axial accelerometer telemetry using Discrete Hidden Markov Models (HMM), Baum-Welch parameter estimation with convergence checks, and Viterbi state sequence decoding.

---

## 📌 Project Overview

Human body motion produces continuous, non-stationary temporal acceleration series governed by biomechanical constraints and inertia. Unlike static classifiers that treat windows in isolation, **Hidden Markov Models (HMMs)** capture temporal persistence and transitions across postures. 

This repository satisfies the requirements for **Formative Assessment 2 (Individual)** in Machine Learning:
- **Zero Third-Party HMM Wrappers:** Full from-scratch implementation of scaled Forward-Backward trellis algorithms, Baum-Welch (Expectation-Maximization) parameter estimation, and Viterbi decoding.
- **Dataset Size:** 52 distinct recordings (13 takes $\times$ 4 activities, each 8–10 seconds at 50 Hz), totaling over 1 minute and 40 seconds per activity.
- **Strict Data Isolation:** 48 takes reserved for training (`raw_recordings/`), and 4 takes withheld completely for out-of-sample evaluation (`unseen_test/`).
- **Rubric Stopping Criterion:** Log-likelihood stopping threshold $|\Delta LL| < 10^{-4}$ enforced during Baum-Welch parameter optimization.
- **Evaluation Performance:** Achieved **83.33% Overall Accuracy** on completely unseen dynamic recordings.

---

## 🎥 Video Verification Evidence

To ensure academic integrity and verify authentic individual data acquisition, four full-body verification videos were recorded showing the researcher holding the recording device in real time:

| Activity | Verification Video Link

| **Still** | [Video Link](https://youtube.com/shorts/jdab_ihb6dA)
| **Standing** | [Video Link](https://youtube.com/shorts/hF6j6BZJ1l0)
| **Walking** | [Video Link](https://youtube.com/shorts/tP7lS637AFM)
| **Jumping** | [Video Link](https://youtube.com/shorts/fWmKLqMak8w) 


---

## 📂 Repository Structure

```text
HMM-Human-Activity-Recognition/
├── dataset/
│   └── (Optional merged CSV exports)
├── raw_recordings/                # 48 Training recordings (12 per activity)
│   ├── still_01/ ... still_12/
│   ├── stand_01/ ... stand_12/
│   ├── walk_01/  ... walk_12/
│   └── jump_01/  ... jump_12/
├── unseen_test/                   # 4 Withheld unseen evaluation recordings
│   ├── still_13/
│   ├── stand_13/
│   ├── walk_13/
│   └── jump_13/
├── figures/                       # Saved plots & heatmaps
│   ├── transition_heatmap.png
│   └── decoded_sequence_plot.png
├── HMM_Activity_Recognition.ipynb # Complete Jupyter Notebook pipeline
├── evaluation_metrics.csv        # Sensitivity, specificity, and accuracy table
├── confusion_matrix.csv          # 4x4 evaluation confusion matrix
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies
```

---

## ⚙️ Mathematical Methodology & Pipeline

### 1. Preprocessing & Resampling
- **Rotational Invariance:** Calculates total acceleration magnitude:
  $$m(t) = \sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2}$$
- **50 Hz Spline Interpolation:** Handles non-uniform timestamp intervals and nanosecond clock exports, interpolating to a uniform grid ($\Delta t = 0.02\text{ s}$).

### 2. Multi-Domain Feature Extraction
Signals are partitioned into sliding windows ($W = 2.0\text{ s}$, $50\%$ overlap, step size $S = 1.0\text{ s}$):
1. **Variance ($\sigma^2$):** $\frac{1}{W}\sum [m(t) - \mu]^2$ (Movement intensity)
2. **Root Mean Square (RMS):** $\sqrt{\frac{1}{W}\sum m(t)^2}$ (Total energy & static gravity)
3. **Dominant Frequency ($f_{\text{dom}}$):** $\operatorname{argmax}_f |X(f)|$ via FFT on detrended signal (Cadence identification)
4. **Spectral Energy ($E_{\text{spec}}$):** $\frac{1}{N_f}\sum |X(f)|^2$ (Ballistic vs. smooth gait discrimination)

### 3. Normalization & Discretization
- Standardized via training Z-Score parameters: $z = \frac{x - \mu}{\sigma}$.
- Continuous feature energies are quantized into $K=8$ quantile discrete observation tokens ($v_k \in \{0, 1, \dots, 7\}$).

### 4. Baum-Welch Parameter Estimation
- Employs scaled forward ($\alpha_t$) and backward ($\beta_t$) variables to eliminate numerical underflow.
- Convergence stopping criterion: $|\Delta LL| = |LL_k - LL_{k-1}| < 10^{-4}$.

### 5. Viterbi Decoding
- Recovers optimal hidden state trajectory $q_{1:T}^*$ in log-space on unseen data:
  $$V_t(j) = \max_i [V_{t-1}(i) + \ln A_{ij}] + \ln B_j(O_t)$$

---

## 📊 Empirical Results & Metrics

### Confusion Matrix on Unseen Data ($N = 30$ Windows)

```text
Actual \ Predicted   Still (S0)   Stand (S1)   Walk (S2)   Jump (S3)
Still (S0)               5            2            0           0
Stand (S1)               1            6            0           0
Walk (S2)                0            0            8           0
Jump (S3)                0            1            1           6
```

### Classification Metrics

| Activity State | Windows Tested | Sensitivity (Recall) | Specificity | State Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Still** | 7 | **0.714** | 0.957 | **83.33%** |
| **Standing** | 7 | **0.857** | 0.870 | **83.33%** |
| **Walking** | 8 | **1.000** | 0.955 | **83.33%** |
| **Jumping** | 8 | **0.750** | **1.000** | **83.33%** |

#### Key Insights
- **Walking (100% Sensitivity):** Zero false negatives; rhythmic harmonic peaks clearly distinct from static postures.
- **Jumping (100% Specificity):** Zero false positives; massive peak variance prevents other actions from being labeled as jumping.
- **Still vs. Standing:** Minor boundary confusion attributable to zero-dynamic-variance overlap between phone resting on a table vs. upright standing in a pocket.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- VS Code or Google Colab

### Installation
Clone the repository and install required scientific computing packages:
```bash
git clone https://github.com/uyvette/HMM-Human-Activity-Recognition.git
cd HMM-Human-Activity-Recognition
pip install -r requirements.txt
```

*(Or run directly: `pip install numpy pandas matplotlib seaborn jupyter`)*

### Running the Pipeline
Open the notebook in VS Code or JupyterLab:
```bash
jupyter notebook HMM_Activity_Recognition.ipynb
```
Run all cells sequentially (`Cell 1` through `Cell 5`) to execute the pipeline, check convergence, and render all transition heatmaps and state trajectory plots.

---

## 👩‍💻 Author
- **Student Name:** Yvette Uwimpaye
- **Department:** Software Engineering, African Leadership University
- **Assessment:** Formative Assessment 2 (Individual Submission)
- **Academic Year:** 2026
