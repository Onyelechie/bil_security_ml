# bil_security_ml

## On-Device Intrusion Detection and False Alarm Filtering

This project is part of COMP 4560: Industrial Project (Winter 2026) in collaboration with BIL Security.

### Project Overview

Security monitoring systems often generate false alarms due to environmental factors such as weather, vegetation, or animals. This project develops an on-device alarm filtering solution that analyzes live camera feeds to detect meaningful intrusion events (presence of people or vehicles) while filtering out non-critical motion.

The system operates on constrained hardware (Windows PCs with i5/i7 processors and ~4GB RAM) and supports multiple simultaneous camera feeds accessed via RTSP.

### Current Status (January 2026)

**Completed:**
- Event-driven detection pipeline (TCP listener, frame buffer, clip extraction)
- Model benchmarking at native HD resolution (1280x720)
- Detection models: YOLOv8n (recommended), YOLOv8s, MobileNet-SSD
- Motion filtering for false alarm reduction
- PySide6 GUI with test panel, zone editor, and alert management
- User-defined monitoring zones with polygon drawing

**Key Finding:** YOLOv8n outperforms MobileNet-SSD at HD resolution (20.4 FPS vs 15.5 FPS) with better detection accuracy.

### Architecture

The system uses an **event-driven clip analysis** approach rather than real-time streaming:

```
Motion Event (TCP) -> Extract 7s clip -> Analyze ~35 frames -> Alert decision
```

This allows accurate YOLO models to run on CPU without requiring a GPU.

### Objectives

- Investigate computer vision and AI techniques for on-device intrusion detection
- Filter false positives caused by weather, animals, or vegetation
- Support configurable sensitivity and monitoring zones
- Evaluate performance under hardware constraints

### Team Members

- Stephen Ugbah
- Bhavik Jain
- Subhash Yadav
- Ebere Onyelechie

### Project Structure

```
bil_security_ml/
├── src/
│   ├── detect/          # Detection models (YOLOv8, MobileNet, motion)
│   ├── events/          # TCP event listener
│   ├── pipeline/        # Frame buffer, clip extraction
│   └── gui/             # PySide6 frontend
├── docs/                # Research documentation
├── models/              # Model weights
├── data/                # Test videos
└── main.py              # Application entry point
```

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the GUI
python main.py
```

### Model Recommendations

| Model | FPS @ 720p | Use Case |
|-------|------------|----------|
| **YOLOv8n** | 20.4 | RECOMMENDED - Best speed + accuracy |
| YOLOv8s | 13.7 | Maximum accuracy needed |
| MobileNet | 15.5 | Legacy systems only |

### Methodology

The project follows a structured approach:

1. **Research and Setup** (Jan 6 - Jan 24, 2026): Initial research on intrusion detection techniques, setup development environment
2. **Research (continued)** (Jan 25 - Feb 7, 2026): Evaluate AI models and finalize architecture
3. **Prototype Development** (Feb 8 - Feb 28, 2026): Implement detection pipeline and performance profiling
4. **Evaluation and Refinement** (Mar 1 - Mar 21, 2026): Test with real data and optimize
5. **Finishing Touches** (Mar 22 - Apr 6, 2026): Finalize documentation and presentation

### Deliverables

- Working proof-of-concept system
- Open-source source code
- Technical documentation
- Final presentation

### Technologies

- Python 3.13
- OpenCV 4.x for video processing
- YOLOv8 (ultralytics) for object detection
- PySide6 for GUI
- RTSP stream processing (pending camera access)

### Technical Notes

- Motion events sent via TCP from existing BIL Security software
- Cameras must be ONVIF Profile S or T compliant:
  - [ONVIF Profile S](https://www.onvif.org/profiles/profile-s/)
  - [ONVIF Profile T](https://www.onvif.org/profiles/profile-t/)

### License

This project is released as open source.