# bil_security_ml

## On-Device Intrusion Detection and False Alarm Filtering

This project is part of COMP 4560: Industrial Project (Winter 2026) in collaboration with BIL Security.

### Project Overview

Security monitoring systems often generate false alarms due to environmental factors such as weather, vegetation, or animals. This project develops an on-device alarm filtering solution that analyzes live camera feeds to detect meaningful intrusion events (presence of people or vehicles) while filtering out non-critical motion.

The system operates on constrained hardware (Windows PCs with i5/i7 processors and ~4GB RAM) and supports multiple simultaneous camera feeds accessed via RTSP.

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

### Methodology

The project follows a structured approach:

1. **Research and Setup** (Jan 6 – Jan 24, 2026): Initial research on intrusion detection techniques, setup development environment
2. **Research (continued)** (Jan 25 – Feb 7, 2026): Evaluate AI models and finalize architecture
3. **Prototype Development** (Feb 8 – Feb 28, 2026): Implement detection pipeline and performance profiling
4. **Evaluation and Refinement** (Mar 1 – Mar 21, 2026): Test with real data and optimize
5. **Finishing Touches** (Mar 22 – Apr 6, 2026): Finalize documentation and presentation

### Deliverables

- Working proof-of-concept system
- Open-source source code
- Technical documentation
- Final presentation

### Technologies

- Computer Vision
- Machine Learning / AI
- RTSP stream processing
- On-device inference on constrained hardware

### Development Setup

#### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

#### Installation
```bash
# Clone the repository
git clone https://github.com/Onyelechie/bil_security_ml.git
cd bil_security_ml

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/Mac:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Running the Server
```bash
# Set Python path for src/ layout
# On Windows PowerShell:
$env:PYTHONPATH = "$PWD\src"
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"

# Run the server
python -m uvicorn server.main:app --reload --port 8000
```

#### Running Tests
```bash
# Set Python path for src/ layout
# On Windows PowerShell:
$env:PYTHONPATH = "$PWD\src"
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"

# Run tests
python -m pytest
```

### Technical Notes

- "Motion events" (from the problem statement) can be sent from our existing software via TCP.
- Ideally, supported security cameras should be compliant with ONVIF Profile S and T standards:
  - [ONVIF Profile S](https://www.onvif.org/profiles/profile-s/)
  - [ONVIF Profile T](https://www.onvif.org/profiles/profile-t/)
- These standards define connection protocols for streaming video, including RTSP.
- Any cameras used will always follow one of these two profiles.

### License

This project is released as open source.