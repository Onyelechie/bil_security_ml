# Model Accuracy Evaluation Report

## Objective

The purpose of this evaluation is to quantify the detection accuracy and localization precision of multiple object detection model families (YOLO, SSD, and EfficientDet) on a specialized security dataset. The dataset consists of 53 manually labeled frames extracted from high-resolution CCTV footage.

## Evaluation Metrics

### Mean Average Precision at $IoU=0.50$ ($mAP_{50}$)

$mAP_{50}$ measures the model's ability to correctly identify and classify objects given an Intersection over Union (IoU) threshold of 0.50. This metric serves as a primary indicator of object detection sensitivity.

### Mean Average Precision across $IoU=0.50:0.95$ ($mAP_{50-95}$)

The $mAP_{50-95}$ metric averages the precision across multiple IoU thresholds in the range [0.50, 0.95]. This metric provides a stringent assessment of the model's localization accuracy and bounding box regression quality.

---

## Comparative Performance Results

The following table summarizes the quantitative results obtained during the benchmarking session.

| Model Family | Variant | $mAP_{50}$ | $mAP_{50-95}$ | $mAP_{75}$ |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8** | **Small (s)** | **0.4276** | **0.3387** | **0.3656** |
| YOLOv8 | Nano (n) | 0.3597 | 0.2465 | 0.2621 |
| YOLOv5 | Nano (n) | 0.3388 | 0.2461 | 0.2840 |
| SSD | MobileNet V3 | 0.1308 | 0.0734 | 0.0682 |
| EfficientDet | D0 | 0.0699 | 0.0281 | 0.0162 |

---

## Technical Analysis

### YOLO Family Performance

The YOLOv8-Small model demonstrated the highest performance across all primary metrics. Its superior localization ($mAP_{75}$ of 0.3656) suggests a robust feature extraction capability specifically for objects at the scales present in CCTV imagery. The Nano variants (YOLOv8 and YOLOv5) provide a high-efficiency alternative, maintaining reasonable accuracy while minimizing computational overhead.

### Underperformance of SSD and EfficientDet

Both SSD-MobileNet and EfficientDet-D0 exhibited significant performance degradation on this dataset. The low $mAP_{50}$ scores indicate a consistent failure to generalize to the specific domain characteristics of the security footage, potentially due to architectural limitations in handling small object detection or domain-specific lighting conditions.

## Conclusion and Recommendations

Based on the empirical evidence, **YOLOv8-Small** is the recommended model for deployments requiring maximum detection reliability. For edge environments with constrained compute resources, **YOLOv8-Nano** is recommended as a high-performance alternative.

The benchmarking environment can be reproduced using the following execution command:

```bash
PYTHONPATH=. ./venv/bin/python accuracy/eval_accuracy.py --dataset accuracy/labeled_data --models all
```

---

## Dataset Setup and Reproduction

The ground-truth dataset used for this evaluation is currently hosted externally (e.g., Google Drive / Discord) due to size and licensing. To reproduce these results:

1. **Source**: Download the labeled datasets from Google Drive:
   - [Model Accuracy Dataset](https://drive.google.com/drive/folders/1ARxKuHc-6C9E5mvKTezLFmP4u3ckL4jJ?usp=drive_link)
   - [Labeled Data (YOLO format)](https://drive.google.com/drive/folders/1reHUtU5yrLBQcFvxbP0RIDaCNJ1NQFci?usp=drive_link)
2. **Structure**: Place the extracted folders so the following structure exists:
   - `accuracy/labeled_data/images/` (.jpg files)
   - `accuracy/labeled_data/labels/` (.txt files)
3. **Negative Samples (Verified)**: Images present in the `images/` directory that **do not** have a corresponding `.txt` file in the `labels/` directory are intentional **negative samples** (background frames with no objects).

### Verified Negative Samples

There are 53 total images but only 50 have annotations. The following 3 frames have been confirmed as intentional negatives (empty background frames):

- `C1LowRes - Human_frame_81.jpg` (ID: 14)
- `C2LowRes - Car_frame_108.jpg` (ID: 31)
- `C2LowRes - Car_frame_135.jpg` (ID: 35)
