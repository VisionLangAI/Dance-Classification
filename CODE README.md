# Fine-Grained Dance Classification using Hybrid LeViT + CNN + Grad-CAM

## 📌 Project Overview

This project presents a hybrid deep learning framework for fine-grained
traditional dance classification using a combination of Convolutional
Neural Networks (CNNs) and LeViT (Lightweight Vision Transformer). The
model integrates Grad-CAM-based explainability to visualize
class-discriminative regions and analyze feature learning across network
depths.

The framework is designed to: - Capture local spatial features using CNN
layers - Model global dependencies using Transformer attention - Provide
visual interpretability using Grad-CAM - Maintain computational
efficiency for real-time and edge deployment

------------------------------------------------------------------------

## 📂 Dataset Description

-   Total Images: 763 (original)
-   Classes:
    -   Sword Dance
    -   Fan Dance
    -   Dragon Dance
-   Data augmentation applied:
    -   Horizontal flipping
    -   Rotation
    -   Brightness adjustment
    -   Width/Height shift
    -   Contrast modification

------------------------------------------------------------------------

## 🏗 Model Architecture

1.  Input Image Layer (224×224×3)
2.  CNN Backbone (Conv Blocks + BatchNorm + ReLU)
3.  Channel Attention Module
4.  Patch Embedding Layer
5.  LeViT Transformer Blocks (Self-Attention + MLP)
6.  Global Feature Aggregation
7.  Gated Fusion Layer
8.  Fully Connected Layers
9.  Softmax Output Layer
10. Grad-CAM Explainability Module

------------------------------------------------------------------------

## ⚙️ Training Configuration

  Parameter           Value
  ------------------- ---------------------------
  Optimizer           Adam
  Learning Rate       Tuned (Adaptive Schedule)
  Batch Size          Optimized for Stability
  Activation          ReLU
  Output Activation   Softmax
  Regularization      Dropout + L2
  Early Stopping      Enabled
  Gradient Clipping   Applied

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   Grad-CAM Depth-wise Analysis
-   Activation Overlap (IoU)
-   Region Consistency
-   Foreground--Background Activation Ratio

------------------------------------------------------------------------

## 🔍 Explainability Analysis

Grad-CAM visualizations were performed at: - Early Convolution Layers -
Intermediate Layers - Final Semantic Layers - With/Without Attention
Blocks - With/Without Transformer Blocks

Quantitative explainability metrics were computed to validate visual
observations.

------------------------------------------------------------------------

## 💻 Computational Efficiency

-   GPU Memory Usage: \~1.74GB (16% utilization)
-   Suitable for edge and mobile deployment

------------------------------------------------------------------------

## 🚀 How to Run

1.  Install dependencies: pip install torch torchvision torchcam
    opencv-python matplotlib

2.  Run the training script or Jupyter Notebook.

3.  Execute Grad-CAM analysis for visualization.

4.  Evaluate metrics and generate explainability tables.

------------------------------------------------------------------------

## ⚠️ Limitations

-   Dataset size is limited (763 images)
-   Only three dance categories included
-   Image-based classification (no temporal modeling)
-   Mild overfitting observed after 30 epochs

Future work includes dataset expansion and spatiotemporal modeling.

------------------------------------------------------------------------

## 📜 Citation

If you use this work, please cite the corresponding research paper.
