# Model Accuracy Evaluation Suite

This directory contains the tools and reports for quantifying object detection accuracy (mAP) on security-specific datasets.

## 🏁 Comparative Accuracy Results

### Production Models (Site-Tuned + OpenVINO)
| Model Family | Variant | $mAP_{50}$ | $mAP_{50-95}$ | $mAP_{75}$ |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8** | **Small-Site** | **0.7209** | **0.6169** | **0.7050** |
| YOLOv8 | Nano-Site | 0.3020 | 0.2199 | 0.2448 |

### Baseline Models (Standard Weights)
| Model Family | Variant | $mAP_{50}$ | $mAP_{50-95}$ | $mAP_{75}$ |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8** | **Small (s)** | **0.3398** | **0.2872** | **0.3094** |
| YOLOv5 | Nano (n) | 0.3020 | 0.2399 | 0.2621 |
| YOLOv8 | Nano (n) | 0.3020 | 0.2199 | 0.2448 |
| SSD | MobileNet V3 | 0.0934 | 0.0438 | 0.0318 |
| EfficientDet | D0 | 0.0538 | 0.0179 | 0.0058 |

---

## 🛠️ How to Run

The evaluation script supports two modes to match the `benchmark_suite.py` logic.

### 1. Baseline Evaluation
To evaluate standard COCO-pretrained models from the `benchmark/` folder:
```bash
python accuracy/eval_accuracy.py
```

### 2. Production Evaluation
To evaluate site-tuned models from the `production_model/` folder:
```bash
python accuracy/eval_accuracy.py --production
```

### Reproducibility Parameters
- **`--dataset`**: Path to the YOLO dataset (default: `accuracy/labeled_data/val`).
- **`--models`**: Comma-separated list of models to evaluate.
- **`--input-size`**: Image resolution for inference (default: `640`).

---

## 📂 Dataset Structure

The ground-truth dataset must follow the YOLO format:
- `accuracy/labeled_data/images/` (.jpg files)
- `accuracy/labeled_data/labels/` (.txt files)

> [!NOTE]
> Images in `images/` without a corresponding `.txt` in `labels/` are treated as **intentional negative samples** (background images where the model should find zero objects).
>
> Note: This repo ignores `.jpg`/`.png` files by default. If you need to track images, update `.gitignore` or provide the image set out-of-band.

### Confirmed Negative Samples
The current test set includes confirmed background frames to verify low false-positive rates:
- `C1LowRes - Human_frame_81.jpg`
- `C2LowRes - Car_frame_108.jpg`
- `C2LowRes - Car_frame_135.jpg`
