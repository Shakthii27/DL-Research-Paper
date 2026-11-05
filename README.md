# MedMamba: Comparative Analysis of Deep Learning Models on PathMNIST

## Overview

This project implements and evaluates **MedMamba**, a hybrid deep learning architecture combining Convolutional Neural Networks (CNNs) and Selective State Space Models (SSMs), for medical image classification.

The primary objective is to investigate the effectiveness of MedMamba in classifying histopathological images and compare its performance against four widely used deep learning architectures:

- ResNet18
- DenseNet
- Vision Transformer (ViT-Tiny)
- Swin Transformer (Swin-Tiny)

All five models are trained and evaluated using the same dataset split, training configuration, and evaluation methodology to ensure a consistent comparison.

## Dataset

**PathMNIST** from the MedMNIST v2 collection is used for this experiment.

PathMNIST consists of histopathological images of colorectal cancer tissue, categorized into nine distinct tissue classes.

| Property | Description |
|---|---|
| Dataset | PathMNIST |
| Source | MedMNIST v2 |
| Image dimensions | 28 × 28 |
| Image channels | 3 (RGB) |
| Number of classes | 9 |
| Training samples used | 5,000 (stratified subset) |
| Validation set | Official MedMNIST validation split |
| Testing set | Official MedMNIST test split |

A stratified sampling strategy is used to select 5,000 training samples while preserving the original class distribution.

The validation and test sets are retained from the official dataset splits.

## Model Architectures

### 1. MedMamba

MedMamba is a hybrid architecture that combines convolutional feature extraction with selective state space modeling.

The implementation uses an SS-Conv-SSM block, where the input channels are divided into two branches:

- **SSM Branch:** Processes features using a two-dimensional selective state space mechanism (SS2D).
- **Convolution Branch:** Extracts local spatial features through convolutional layers.

The outputs are concatenated, passed through channel shuffle, and combined with a residual connection.

#### Key architectural components

- **SS2D:** Implements selective state space modeling across four scanning directions.
- **Cross-Scan Mechanism:** Processes features through row-wise, column-wise, and their reversed scanning directions.
- **Depthwise Convolution:** Captures local spatial information before state space processing.
- **Channel Shuffle:** Enables information exchange between the convolution and SSM branches.
- **Residual Connections:** Improve information flow and training stability.
- **Hierarchical Feature Extraction:** Uses three stages with feature resolutions of 14 × 14, 7 × 7, and 4 × 4.

The selective scan uses input-dependent parameters (dt, B, and C) and a 16-dimensional state.

An equivalent parallel matrix formulation is used to compute the selective scan on the GPU without explicitly iterating through individual tokens using a Python loop.

#### MedMamba configuration

| Parameter | Value |
|---|---|
| Dimensions | (64, 128, 256) |
| Stage depths | (2, 4, 2) |
| State dimension | 16 |
| Number of heads | 2 |
| Drop path | 0.1 |
| Head dropout | 0.1 |
| Number of classes | 9 |

### 2. ResNet18

ResNet18 is a residual CNN architecture that uses skip connections to facilitate deeper network training.

The original input stem is modified to accommodate 28 × 28 images using a 3 × 3 convolution with stride 1 and no initial max-pooling operation.

### 3. DenseNet

DenseNet uses dense connectivity, where each layer receives feature maps from preceding layers.

A compact DenseNet configuration is implemented with modifications to its initial convolution and pooling layers to support smaller input images.

### 4. ViT-Tiny

Vision Transformer divides an image into patches and processes them as a sequence using transformer encoder layers.

The implementation uses:

- Patch size: 4 × 4
- Embedding dimension: 192
- Transformer depth: 4
- Attention heads: 3
- Learnable positional embeddings
- Classification token

### 5. Swin-Tiny

Swin Transformer uses shifted-window self-attention to capture local and hierarchical image representations.

The implementation uses 2 × 2 patches, 7 × 7 attention windows, and two hierarchical stages.

## Training Methodology

To maintain consistency, the same general training methodology is applied to all five architectures.

### Training configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 25 |
| Batch size | 128 |
| Optimizer | AdamW |
| Weight decay | 0.05 |
| Label smoothing | 0.1 |
| EMA decay | 0.99 |
| Learning rate (CNNs and MedMamba) | 0.002 |
| Learning rate (Transformers) | 0.001 |
| Learning rate scheduler | OneCycleLR |
| Gradient clipping | 1.0 |
| Mixed precision | FP16 (when CUDA is available) |
| Random seed | 42 |

### Data Augmentation

GPU-based data augmentation is applied during training.

The augmentation techniques include:

- Horizontal and vertical flips
- Random 90-degree rotations and transpositions
- Random shifts and zoom transformations
- Brightness adjustment
- Contrast adjustment
- Saturation adjustment
- Per-channel colour variation

Images are normalized to the range [-1, 1].

### Additional Training Techniques

**1. Exponential Moving Average (EMA)**

EMA maintains a smoothed version of model weights, which is used for validation and final evaluation.

**2. Learning Rate Scheduling**

OneCycleLR is used with a warm-up phase followed by cosine annealing.

**3. Label Smoothing**

Label smoothing with a value of 0.1 is used to reduce overconfidence during classification.

**4. Mixed Precision Training**

FP16 autocasting and gradient scaling are enabled when a CUDA GPU is available.

**5. Gradient Clipping**

Gradient norms are clipped to 1.0 to improve training stability.

## Evaluation Metrics

The models are evaluated on the official PathMNIST test split.

The following metrics are calculated:

- **Accuracy:** Proportion of correctly classified images.
- **Macro Precision:** Average precision across all nine classes.
- **Macro Recall:** Average recall across all nine classes.
- **Macro F1-score:** Unweighted average of the F1-scores across classes.
- **Weighted F1-score:** F1-score weighted according to class support.
- **Test Loss:** Negative log-likelihood of predicted probabilities.

### Test-Time Augmentation (TTA)

Eight different flip and rotation transformations are applied during inference.

The predicted probability distributions from all eight transformations are averaged to obtain the final prediction.

Both accuracy without TTA and accuracy with TTA are reported for comparison.

## Visualizations and Analysis

The notebook generates the following visualizations:

1. Accuracy and Macro F1 comparison across all five models.
2. Validation accuracy comparison across epochs.
3. Individual training and validation loss curves.
4. Individual training and validation accuracy curves.
5. Confusion matrices for each model.
6. Classification reports containing precision, recall, and F1-score for each class.

These visualizations help analyze model convergence, classification performance, and class-wise prediction behaviour.

## Project Structure

```text
MedMamba-PathMNIST/
│
├── MedMamba_FINAL(3).ipynb
│
└── pathmnist_results/
    ├── MedMamba_best.pt
    ├── ResNet18_best.pt
    ├── DenseNet_best.pt
    ├── ViT-Tiny_best.pt
    ├── Swin-Tiny_best.pt
    │
    ├── final_results.csv
    ├── accuracy_f1_comparison.png
    ├── val_accuracy_all_models.png
    │
    ├── MedMamba_history.csv
    ├── ResNet18_history.csv
    ├── DenseNet_history.csv
    ├── ViT-Tiny_history.csv
    ├── Swin-Tiny_history.csv
    │
    ├── *_curves.png
    ├── *_confusion_matrix.png
    └── *_classification_report.txt
```

The results directory is generated when the notebook is executed.

## Requirements

The project uses Python and PyTorch.

Main dependencies:

```bash
pip install medmnist
pip install torch torchvision
pip install numpy pandas matplotlib seaborn scikit-learn
```

Google Colab with GPU acceleration is recommended for faster training.

## How to Run

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another compatible GPU).
3. Execute the notebook cells sequentially.
4. The PathMNIST dataset will be downloaded automatically through MedMNIST.
5. All five models will be trained using the specified configuration.
6. Test-set evaluation, performance comparisons, and visualizations will be generated.
7. The experiment outputs will be saved in `/content/pathmnist_results/`.
8. The notebook creates a ZIP archive containing the generated results.

## Reproducibility

The experiment uses a fixed random seed of 42 for Python, NumPy, and PyTorch.

All models use the same training subset, validation set, test set, batch size, epoch count, augmentation strategy, and evaluation procedure.

However, the learning rates are architecture-specific, and the model architectures have different parameter counts and computational requirements.

## Objective

The primary objective of this experiment is to compare the performance of a hybrid CNN-SSM architecture against conventional CNN-based and Transformer-based architectures for histopathological image classification.

The experiment investigates whether combining local spatial feature extraction with selective state space modeling can provide an effective alternative to conventional approaches for medical image classification.

---

**Frameworks:** PyTorch, Torchvision, MedMNIST, Scikit-learn

**Dataset:** PathMNIST (MedMNIST v2)

**Task:** Multiclass Medical Image Classification

**Models:** MedMamba, ResNet18, DenseNet, ViT-Tiny, Swin-Tiny
