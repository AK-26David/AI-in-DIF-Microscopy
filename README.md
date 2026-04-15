# 🧠 AIBD Classification using BMZ-Aware Deep Learning

## 📌 Overview

This project presents a **BMZ-aware deep learning framework** for the classification of **Autoimmune Blistering Diseases (AIBDs)** from Direct Immunofluorescence (DIF) images.

The model integrates:

* Spatial reasoning via **Basement Membrane Zone (BMZ) extraction**
* Physics-inspired **distance encoding**
* **Attention mechanisms**
* **Prototype-based learning (BCPL)**

It jointly performs:

> ✅ Disease Detection (Negative vs Disease)
> ✅ Disease Subtype Classification (PV, PF, BP, EBA, Vasculitis)

---

## 🧩 Problem Statement

DIF image classification is challenging due to:

* Subtle fluorescence patterns
* High inter-class similarity (e.g., PF vs PV)
* Severe class imbalance
* Lack of spatial grounding in standard CNNs

---

## 🚀 Key Contributions

* 📍 BMZ-aware spatial modeling
* 📏 Signed **distance-from-BMZ encoding**
* 🎯 Attention-based fluorescence localization
* 🧬 Prototype-based feature learning (BCPL)
* 🔄 Joint detection + classification framework

---

## 🏗️ System Architecture

### 🔹 Pipeline Overview

```
Input DIF Image
        ↓
Preprocessing (Resize, Normalize)
        ↓
BMZ Detection
        ↓
BMZ Mask + Distance Map
        ↓
Feature Augmentation (4-channel input)
        ↓
ResNet Encoder + BMZ Attention
        ↓
Feature Projection
        ↓
BMZ Profile Encoding (Auxiliary)
        ↓
Feature Fusion
        ↓
Prototype Layer (BCPL)
        ↓
Final Classification (6 classes)
```

---

### 🔹 Model Architecture

```
              ┌──────────────────────┐
              │   Input Image (RGB)  │
              └─────────┬────────────┘
                        │
                + Distance Map
                        │
                        ▼
            ┌────────────────────┐
            │  Image Encoder     │
            │  (ResNet + Attn)   │
            └─────────┬──────────┘
                      │
              Feature Vector (256)
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
BMZ Profile Encoder         Distance Encoding
        │                           │
        └─────────────┬─────────────┘
                      ▼
              Feature Fusion (320D)
                      ▼
            Prototype Layer (BCPL)
                      ▼
              Classification Output
```

---

## 🧠 Methodology

### 1. BMZ Extraction

* Gradient-based localization of fluorescence boundary
* Generates BMZ mask

### 2. Distance Map Encoding

* Signed distance transform from BMZ
* Encodes spatial fluorescence distribution

### 3. Attention Mechanism

* Focuses model on biologically relevant regions

### 4. Prototype Learning (BCPL)

* Learns class-wise feature centers
* Improves interpretability and separation

---

## 📊 Dataset

Classes:

* NEGATIVES
* PV (Pemphigus Vulgaris)
* PF (Pemphigus Foliaceus)
* BP (Bullous Pemphigoid)
* EBA (Epidermolysis Bullosa Acquisita)
* Vasculitis

### Distribution (imbalanced)

* BP / PV → dominant
* PF / EBA → rare

---

## 📈 Results

### 🔹 Overall Performance

| Metric          | Value           |
| --------------- | --------------- |
| Accuracy        | **74%**         |
| Weighted F1     | **0.77**        |
| Binary Accuracy | **~99% (peak)** |

---

### 🔹 Class-wise Performance

| Class      | Precision | Recall | F1   |
| ---------- | --------- | ------ | ---- |
| NEGATIVES  | 0.32      | 0.97   | 0.49 |
| PV         | 0.87      | 0.72   | 0.79 |
| PF         | 0.16      | 0.58   | 0.26 |
| BP         | 0.92      | 0.77   | 0.84 |
| EBA        | 0.41      | 0.28   | 0.33 |
| Vasculitis | 0.84      | 0.75   | 0.79 |

---

## 🔍 Key Insights

### ✅ Strengths

* Excellent **disease detection (~99%)**
* Strong performance on dominant classes (PV, BP, Vasculitis)
* Effective spatial reasoning via BMZ modeling

---

### ⚠️ Challenges

#### PF vs PV confusion

* Subtle depth differences in fluorescence
* Clinically difficult problem

#### EBA performance

* Limited samples
* Weak prototype formation

#### NEGATIVE precision

* Model occasionally predicts negative for weak signals

---

## 📊 Visualizations (Recommended)

Include:

* Confusion Matrix
* t-SNE Feature Space
* BMZ Attention Maps
* Prototype Similarity Matrix
* Misclassification Examples

---

## 🧪 Training Details

* Backbone: ResNet18
* Input: 4-channel (RGB + Distance Map)
* Loss:

  * Focal Loss (class-balanced)
  * Prototype Loss
  * Attention Regularization
* Optimizer: Adam (1e-4)

---

## 🔬 Conclusion

The proposed system demonstrates that:

> Incorporating spatial priors such as BMZ structure significantly improves DIF image understanding.

Key takeaway:

* Detection is easier than subtype classification
* Spatial encoding is critical for dermatological imaging

---

## 🚀 Future Work

* Multi-channel IF (IgA, C3 integration)
* Learned BMZ segmentation
* Transformer-based spatial modeling
* Clinical deployment pipeline

---

## 📌 Author

Arnav Karnik - AK26David

---

## ⭐ If you found this useful

Give it a ⭐ and feel free to fork!
