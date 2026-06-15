# Trustworthy AI in Medical Imaging: Adversarial Vulnerabilities and Explainability in Skin Cancer Classification

This repository contains the complete empirical framework, training pipelines, and safety evaluations for my Master's Dissertation exploring the robustness-accuracy boundaries of Deep Learning systems within digital dermatology.

## 📌 Project Overview
As Deep Neural Networks are increasingly integrated into clinical decision-support systems, ensuring their resilience against adversarial manipulation and maintaining semantic transparency is critical. This project implements a robust benchmarking environment using a **ResNet50** backbone trained on the HAM10000 skin lesion dataset to classify 7 distinct diagnostic categories. 

The core research objectives focus on mapping the system's degradation under targeted perturbation regimes and validating structural defense strategies alongside visual explainability metrics.

---

## 🛠️ Research Framework & Timeline

### 🔹 Months 1–2: Dataset Engineering & Baseline Optimization
* Curated and balanced clinical imagery across 7 distinct lesion classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
* Optimized a baseline ResNet50 classifier to a peak clean validation accuracy of **87.67%**.

### 🔹 Month 3: Adversarial Vulnerability Profiling
* Developed white-box threat simulation engines implementing **Fast Gradient Sign Method (FGSM)** and **Projected Gradient Descent (PGD)** iterative attacks ($\epsilon = 0.02$).
* Demonstrated total system collapse under PGD optimization (Accuracy dropped from 87.67% to **0.00%**, achieving a **100.00% Attack Success Rate**).

### 🔹 Month 4: Defensive Evaluation
* Engineered and benchmarked Preprocessing-based Denoising Filters (Gaussian Blur) vs. Structural Robust Training adjustments.
* Evaluated the iconic *Robustness vs. Accuracy Trade-off*, successfully mitigating complete collapse by recovering PGD robust training parameters to **44.90%**.

### 🔹 Month 5: Explainability Analysis (Grad-CAM)
* Constructed native PyTorch hook layers to map activation gradients at the terminal convolutional block (`model.layer4[-1]`).
* Visually confirmed that adversarial attacks exploit the model by forcibly shifting its diagnostic attention maps away from primary lesion structures onto healthy skin boundaries and background artifacts.

---

## 📊 Visual Workspace & Empirical Assets

All primary analytical charts, training curves, and validation visual maps generated during this project are organized into dedicated feature development branches:
* 📈 **`feature/baseline-analysis`**: Contains baseline optimization learning curves confirming clean convergence.
* 📊 **`feature/adversarial-benchmarks`**: Contains comparative performance bar charts highlighting system accuracy drops and Attack Success Rates (ASR).
* 👁️ **`feature/explainability-defense`**: Contains side-by-side Grad-CAM heatmaps validating semantic focus shifts alongside defense calibration logs.

---

## 🚀 Environment & Implementation Specs
* **Backbone Architecture:** ResNet50 (Transfer Learning Paradigm)
* **Framework:** PyTorch 2.x / CUDA Acceleration
* **Explainer Engine:** Gradient-weighted Class Activation Mapping (Grad-CAM)
* **Threat APIs:** Native Iterative Fast Gradient Optimization Engine
