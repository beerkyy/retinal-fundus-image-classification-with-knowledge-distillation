**Report:** https://docs.google.com/document/d/1jKA3PTr3ArMirFCTWQaN1hSC8FnfkOHVFhK_yzEubx0/edit?usp=sharing
**Presentation Slides:** https://docs.google.com/presentation/d/1jifaBrqvHRTbLcaaAQ8g99kS4ic0eXQ8/edit?slide=id.p1#slide=id.p1

# 🧠 Retinal Fundus Image Classification with Knowledge Distillation

A lightweight, edge-deployable deep learning pipeline for classifying retinal fundus anomalies (e.g., Diabetic Retinopathy, Glaucoma, Cataract) using **cross-architecture knowledge distillation (KD)** from a Vision Transformer (ViT) to a compact CNN—optimized for deployment on the NVIDIA Jetson Nano.

---

## 📌 Overview

- 🧪 **Teacher**: Vision Transformer (ViT) pretrained with I-JEPA  
- 🧠 **Student**: MobileNetV2/ResNet18 CNN distilled from the teacher  
- 📦 **Distillation Modules**: Partitioned Cross-Attention (PCA), Group-Wise Linear (GL), Adversarial Matching  
- 💻 **Deployment Target**: NVIDIA Jetson Nano

> ⚙️ Achieved **~89% accuracy** with only **2.2M parameters**, retaining ~93.8% of the ViT’s diagnostic power — all deployable on edge hardware.

---

## 🖼️ Sample Architecture

<p align="center">
  <img src="https://user-images.githubusercontent.com/your_image_path.png" width="600"/>
  <br><em>Cross-architecture distillation from ViT to CNN with multi-loss training</em>
</p>

---

## 📁 Repository Structure

| File / Folder                  | Purpose |
|-------------------------------|---------|
| `data_collection_preproc_finetuning.ipynb` | Preprocessing + fine-tuning for teacher |
| `data_loading.py`, `data_preprocessing_v2.ipynb` | Data loading and augmentation |
| `finetune_vit.py`, `finetune_config.py` | Teacher model training config |
| `model.py`, `trainer.py`, `optimizer.py` | Core architecture and training loop |
| `evaluate_student_Jetson.ipynb` | Student model evaluation (Jetson-ready) |
| `evaluator.py`, `utils.py` | Metrics and utility functions |
| `requirements.txt` | Dependencies |
| `README.md` | This file! |

---

## 📊 Results

| Metric        | ViT (Teacher) | Student (CNN) |
|---------------|---------------|----------------|
| Accuracy      | 94.6%         | **89.0%**      |
| Parameters    | 85.8M         | **2.2M**       |
| Deployment    | ❌            | ✅ Jetson Nano |
| F1 (Cataract) | 83.2%         | 85.6%          |
| F1 (Glaucoma) | 71.3%         | 68.7%          |

---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/aniruddhaiyengar/retinal-fundus-image-classification-with-knowledge-distillation.git
cd retinal-fundus-image-classification-with-knowledge-distillation

# Set up environment
pip install -r requirements.txt

# Train the ViT teacher model
python finetune_vit.py

# Train the CNN student using KD
python trainer.py

# Evaluate performance on test set
python evaluator.py
