# yolo11-ph-traffic-detection

YOLO11n multi-class vehicle detection for Philippine urban traffic — comparative hyperparameter study across 10 vehicle classes including jeepney, tricycle, and e-bike.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.19-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project trains and compares three YOLO11n object detection models on a custom Philippine urban traffic dataset. The dataset was built from YouTube footage of Metro Manila traffic and annotated across 10 vehicle classes — including Philippine-specific categories absent from standard datasets like COCO and ImageNet.

---

## Dataset

- **Source:** 3 YouTube videos of Metro Manila traffic (~15 minutes total)
- **Frames extracted:** 483 (at 0.5 fps using OpenCV)
- **Annotated images:** 159 (hybrid manual + AI-assisted via Roboflow Label Assist)
- **After 3× augmentation:** 450 training images
- **Total bounding box annotations:** 6,756
- **Classes (10):** bike, bus, car, e-bike, jeepney, motorcycle, sidecar, tricycle, truck, van
- **Roboflow project:** [PH Traffic Vehicle Classification](https://roboflow.com)

---

## Model Configurations

| Model | Optimizer | Batch | Learning Rate | Epochs |
|-------|-----------|-------|---------------|--------|
| Model 1 | AdamW | 4 | 0.0100 | 25 |
| Model 2 | SGD | 20 | 0.0010 | 30 |
| Model 3 | auto | -1 (auto) | 0.0001 | 40 |

---

## Results

| Model | mAP50 | Precision | Recall | F1 Score |
|-------|-------|-----------|--------|----------|
| Model 1 (AdamW, lr=0.01, ep=25) | 0.0197 | 0.3751 | 0.0193 | 0.0367 |
| Model 2 (SGD, lr=0.001, ep=30) | 0.2153 | 0.2376 | 0.1936 | 0.2134 |
| **Model 3 (auto, lr=0.0001, ep=40)** | **0.2476** | **0.4355** | **0.2169** | **0.2896** |

> Model 3 achieved the best performance across all metrics. All models failed to detect minority classes (jeepney, tricycle, bus, e-bike, bike, sidecar) due to extreme class imbalance.

---

## Weights

Pre-trained weights are available in the [Releases](../../releases) section.

| File | Model | mAP50 |
|------|-------|-------|
| `model1_best.pt` | AdamW, batch=4, lr=0.01, 25 epochs | 0.0197 |
| `model2_best.pt` | SGD, batch=20, lr=0.001, 30 epochs | 0.2153 |
| `model3_best.pt` | Auto, batch=-1, lr=0.0001, 40 epochs | 0.2476 |

---

## Environment

- Google Colaboratory (Pro)
- NVIDIA Tesla T4 GPU (14,913 MiB)
- Python 3.12.12
- PyTorch 2.10.0+cu128
- Ultralytics 8.4.19

---

## Installation
```bash
pip install ultralytics==8.4.19
pip install torch>=2.0.0 opencv-python pyyaml pandas
```

---

## Usage
```python
from ultralytics import YOLO

# Load best model
model = YOLO('model3_best.pt')

# Run inference
results = model.predict('your_image.jpg', imgsz=640)
results[0].show()
```

---

## Repository Structure
```
yolo11-ph-traffic-detection/
├── SA2_PEREZ.ipynb          # Training notebook
├── data.yaml                # Dataset configuration
├── requirements.txt
├── results/
│   ├── model1_adamw/        # Curves, confusion matrix, val predictions
│   ├── model2_sgd/
│   └── model3_auto/
└── weights/                 # See Releases for .pt files
```

---

## Author

**Dharl Russell C. Perez**  
Mapua University — Data Science 3  
Adviser: John Paul Q. Tomas  
Co-author: Bonifacio T. Doma Jr., Ph.D.
