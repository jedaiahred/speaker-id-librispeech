# Speaker Identification on LibriSpeech

**317 Speakers · 97% Test Accuracy · 0.965 Weighted F1**

A speaker identification pipeline built with TensorFlow and trained on the LibriSpeech corpus. The model classifies 10-second audio clips as belonging to one of 317 speakers using MFCC features and a transformer-inspired architecture.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![GPU](https://img.shields.io/badge/GPU-RTX%205060%20Ti-green)

---

## Results

### Single Split

| Metric | Test Set |
|---------------------|----------|
| Accuracy | 96.7% |
| Weighted F1 | 0.965 |
| Weighted Precision | 0.971 |
| Weighted Recall | 0.967 |
| Brier Score | 0.046 |

### Stratified 4-Fold Cross Validation

| Metric | Mean | Std |
|----------------|-------|-------|
| Accuracy | 97.0% | 0.008 |
| Weighted F1 | 0.969 | — |
| Brier Score | 0.043 | 0.011 |

---

## Architecture

A custom subclassed Keras model with a transformer-inspired design:

1. **Permute** — reorients input so attention operates over the time axis.
2. **Layer Normalization** — stabilizes attention inputs.
3. **Multi-Head Causal Attention** (4 heads, key dim 15) — learns temporal dependencies across MFCC frames.
4. **Conv1D** (256 filters, kernel size 128, causal padding) — captures local temporal patterns.
5. **Flatten → Dense (512) → Dropout (0.1) → Dense (1024)** — nonlinear classification head.
6. **Logits → Temperature Scaling → Softmax** — outputs calibrated class probabilities.

Orthogonal kernel regularization is applied to all three dense layers.

---

## Training

| Parameter | Value |
|--------------------------|--------------------------------------|
| Epochs | 20 (early stop at 97.5% val acc) |
| Batch Size | 128 |
| Optimizer | Adam (β₁=0.95, β₂=0.97) |
| Learning Rate | 1e-3, exponential decay (0.75) |
| Loss | Scaled Brier Loss (scale=10,000) |
| Mixed Precision | FP16 compute, FP32 accumulation |
| Regularization | Orthogonal kernel reg, dropout (0.1) |

**Brier loss** is used instead of cross-entropy to directly optimize probabilistic accuracy. The loss is scaled by 10,000 to prevent gradient underflow under FP16 mixed precision.

---

## Probabilistic Calibration

After training, the model achieves high accuracy but produces overconfident probabilities on incorrect predictions. **Temperature scaling** is applied as a post-hoc calibration step: all model weights are frozen, and a single learned temperature parameter is optimized on the validation set for 10 epochs using categorical cross-entropy.

| | Brier (Val) | Brier (Test) |
|---------------------------|-------------|--------------|
| Before temperature scaling | 0.0452 | 0.0467 |
| After temperature scaling | 0.0449 | 0.0464 |

---

## Dataset

[LibriSpeech](https://www.openslr.org/12) — 34,342 audio recordings from 317 speakers, drawn from three subsets:

- `train-clean-100`
- `dev-other`
- `test-other`

### Preprocessing

1. **Amplitude normalization**: Mean-center and scale to [-1, 1]. Signals longer than 10s are truncated before normalization; shorter signals are normalized before zero-padding.
2. **MFCC extraction**: 20 coefficients (n_fft=2048, hop_length=512) via librosa.
3. **Delta features**: First and second-order deltas (width=9) appended to MFCCs, yielding a 60×314 input per recording.
4. **Stratified splits**: 75% train / 20% validation / 5% test, stratified by speaker to handle class imbalance.

---

## Environment

This project was trained on an **NVIDIA RTX 5060 Ti** (Blackwell architecture) using NVIDIA's NGC TensorFlow container. The `.devcontainer` directory provides a reproducible environment for VS Code + Docker.

### Quick Start

1. Clone the repo and open in VS Code.
2. Reopen in the dev container (requires Docker and the NVIDIA Container Toolkit).
3. Run the notebook — it downloads the LibriSpeech subsets automatically.

---

## Repository Structure

```
artifacts/
.devcontainer/
    devcontainer.json
    Dockerfile
README.md
speaker-identification.ipynb
```
