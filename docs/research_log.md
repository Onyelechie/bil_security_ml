# Research Log - On-Device Intrusion Detection

## Project: BIL Security - False Alarm Filtering
## Course: COMP 4560 Industrial Project - Winter 2026

---

## Research Log

### January 21, 2026 - Initial Pipeline Development

**Objective:** Build foundational pipeline components before receiving RTSP camera access.

#### What We Tested

1. **TCP Event Listener Architecture**
   - Async TCP server using Python's `asyncio`
   - JSON message parsing and normalization
   - Event acknowledgment responses

2. **Ring Buffer for Frame Storage**
   - Thread-safe deque-based implementation
   - Time-based frame retention (30s window)
   - Frame extraction around event timestamps

4. **Event Frame Window Extraction**
   - Window: t-2s to t+5s around event time
   - Tested with 75 frames (~5s at 15 FPS)
   - Output: MP4 video clips (~279 KB for 5s clip)

#### What Worked

- ✅ TCP listener successfully receives and parses JSON events
- ✅ Ring buffer correctly stores and retrieves timestamped frames
- ✅ Video clip saving with OpenCV VideoWriter (mp4v codec)
- ✅ End-to-end pipeline: event → frame extraction → MP4 output

#### What Failed / Challenges

- ⚠️ Need to handle case where event arrives before buffer is full (< 2s of history)

#### Decisions Made

| Decision | Reasoning |
|----------|-----------|
| Use asyncio for TCP server | Non-blocking I/O for handling multiple connections, native Python support |
| JSON for event format | Easy to debug, flexible schema, can switch to binary later if needed |
| Ring buffer with time-based cleanup | Efficient memory use, automatic old frame removal |
| Default to 15 FPS | Balance between temporal resolution and CPU/memory cost |
| 30s buffer duration | Covers t-2s to t+5s window with margin for late events |
| MP4 output format | Universal playback, reasonable compression |

#### Next Steps

- [ ] Receive RTSP URL from industry partner
- [ ] Implement RTSP stream reader with reconnection logic
- [ ] Test CPU usage with real video streams
- [ ] Evaluate frame window size impact on detection accuracy

---

## Planned Experiments

| Experiment | Hypothesis | Variables | Metrics | Status |
|------------|------------|-----------|---------|--------|
| RTSP decode CPU cost | Decoding RTSP will be a significant CPU bottleneck | Resolution, codec, FPS | CPU %, decode latency (ms) | 🔜 Waiting for RTSP URL |
| FPS vs accuracy tradeoff | Lower FPS reduces CPU but may miss fast motion | 5, 10, 15, 30 FPS | CPU %, detection recall, false negatives | 🔜 Pending |
| Frame window size impact | Larger window improves context but increases processing | ±2s, ±5s, ±10s | Detection accuracy, clip size, processing time | 🔜 Pending |
| Lightweight model comparison | MobileNet/YOLO-Nano vs full models | Model architecture | mAP, inference time, RAM usage | ✅ Completed |
| Multi-stream scaling | 10 simultaneous streams on i5/i7 | Number of streams | CPU %, memory, dropped frames | 🔜 Pending |
| Motion filtering approaches | Pre-filtering reduces unnecessary inference | Background subtraction, frame diff | True positive rate, CPU savings | ✅ Implemented |

---

### January 25, 2026 - Model Speed Benchmarks & Detection Pipeline

**Objective:** Benchmark detection model speeds and complete the intrusion detection pipeline.

#### Model Speed Benchmark Results

**Test Setup:**
- Test Video: VIRAT_S_010204_05_000856_000890.mp4 (1280x720, VIRAT surveillance dataset)
- Test Frames: 100 frames resized to 640x480
- System: Windows PC (CPU-only inference, no GPU)

| Model | FPS | Avg (ms) | Min (ms) | Max (ms) | Classes | Event-Driven Status |
|-------|-----|----------|----------|----------|---------|---------------------|
| MobileNet-SSD | 79.0 | 12.7 | 10.0 | 28.9 | 21 | ✓ Fastest, lower accuracy |
| **YOLOv8n** | **28.3** | 35.3 | 32.7 | 44.3 | 80 | ⭐ **Close-range cameras** |
| **YOLOv8s** | **13.6** | 73.7 | 70.6 | 86.0 | 80 | ⭐ **Distant cameras** |
| YOLOv8m | 5.9 | 170.1 | 164.3 | 227.1 | 80 | ✓ Slower, better accuracy |
| YOLOv8l | 3.0 | 332.1 | 322.0 | 374.6 | 80 | ✓ Slow, high accuracy |
| YOLOv8x | 2.1 | 486.8 | 477.7 | 553.0 | 80 | ✓ Slowest, best accuracy |

*Note: All models viable for event-driven clip analysis. "Too slow" only applies to real-time streaming.*

#### Key Findings

1. **YOLOv8s recommended for distant/wide-angle cameras**
   - Manual testing showed significantly better detection accuracy than MobileNet
   - 13.6 FPS is sufficient for event-driven clip analysis (see architecture comparison below)
   - 80 classes provides comprehensive object detection
   - Better at detecting small/distant people and vehicles
   - Best for: parking lots, perimeters, wide coverage areas

2. **YOLOv8n recommended for close-range/entry-point cameras**
   - 28.3 FPS - faster analysis (~1.2 seconds per clip)
   - Good accuracy for larger objects (people at doors, vehicles at gates)
   - Best for: doorways, gates, close-range entry points
   - Use when subjects are closer to camera and appear larger in frame

3. **Event-Driven vs Real-Time Streaming: Why YOLO Works on CPU**

   Our architecture analyzes **short clips after motion events**, not continuous live streams.
   This fundamentally changes the hardware requirements:

   | Architecture | Description | FPS Requirement | YOLOv8s Viable? |
   |--------------|-------------|-----------------|-----------------|
   | **Real-Time Streaming** | Process every frame from all cameras continuously | 150 FPS (10 cams × 15fps) | ❌ No (only 13.6 FPS) |
   | **Event-Driven Clips** ⭐ | Analyze 7-second clips when motion detected | ~35 frames per event | ✅ Yes! |

   **Event-Driven Clip Analysis (Our Approach):**
   ```
   Motion Event → Extract 7s clip → Analyze ~35 frames → Result in ~2.6 seconds
   ```

   | Metric | Calculation | Result |
   |--------|-------------|--------|
   | Clip length | t-2s to t+5s | 7 seconds |
   | Frames per clip (15fps) | 7 × 15 | 105 frames |
   | With frame skip (every 3rd) | 105 ÷ 3 | **35 frames** |
   | Analysis time @ 13.6 FPS | 35 ÷ 13.6 | **2.6 seconds** |

   **Why this works for 10 cameras:**
   - Motion events are **sporadic**, not constant (maybe 1-5 per minute across all cameras)
   - Events **queue up** and process sequentially
   - 2.6 seconds per event means ~23 events/minute capacity
   - Even during busy periods, queue catches up quickly
   - **No GPU required** for 10 cameras with YOLOv8s

4. **MobileNet-SSD as fallback option**
   - 79 FPS - only needed for extremely high event volumes
   - Lower accuracy - may miss detections that YOLOv8s catches
   - Use only for legacy hardware or 50+ events/minute scenarios

5. **Larger YOLO models (m/l/x) for maximum accuracy**
   - YOLOv8x at 2.1 FPS: ~17 seconds per clip (35 frames ÷ 2.1 FPS)
   - Still viable for event-driven analysis if accuracy is paramount
   - GPU only needed for real-time streaming (not our use case)

#### GPU Requirements (Real-Time Streaming Only)

**Note: GPU is NOT required for our event-driven architecture.** This section is for reference only if real-time streaming was needed.

| Hardware | YOLOv8x FPS | Real-Time Cameras | Event-Driven Clips |
|----------|-------------|-------------------|-------------------|
| CPU only (i5/i7) | ~2 FPS | ❌ 0 cameras | ✅ Works fine |
| NVIDIA RTX 3060 | ~80-120 FPS | 5-8 cameras | ✅ Overkill |
| NVIDIA RTX 3080 | ~150-200 FPS | 10+ cameras | ✅ Overkill |

#### Model Recommendations by Camera Count

**For Event-Driven Clip Analysis (BIL Security Architecture):**

| Cameras | Recommended Model | Analysis Time/Clip | Events/Minute Capacity | Notes |
|---------|-------------------|-------------------|------------------------|-------|
| 1-10 (distant) | **YOLOv8s** ⭐ | ~2.6 sec | ~23 events | Best for wide/distant views |
| 1-10 (close) | **YOLOv8n** ⭐ | ~1.2 sec | ~50 events | Best for entry points/close-range |
| Mixed setup | YOLOv8s + YOLOv8n | ~1.2-2.6 sec | ~30 events | Configure per camera type |
| High volume | YOLOv8n | ~1.3 sec | ~46 events | Faster, slightly less accurate |
| Legacy HW | MobileNet | ~0.4 sec | ~150 events | Last resort, lower accuracy |

⭐ = Recommended for BIL Security project

**For Real-Time Streaming (NOT our architecture, but for reference):**

| Cameras | CPU-Only | With GPU (RTX 3060+) | Notes |
|---------|----------|----------------------|-------|
| 1-2 | YOLOv8s | YOLOv8x | Best accuracy |
| 3-5 | YOLOv8n or MobileNet | YOLOv8l/x | Balance speed/accuracy |
| 6-10 | MobileNet + frame skip | YOLOv8x | GPU required for YOLO |
| 10+ | MobileNet + aggressive skip | YOLOv8x | Needs RTX 3080+ |

#### Detection Pipeline Test Results

Tested end-to-end pipeline with VIRAT surveillance video:

```
Video: VIRAT_S_010204_05_000856_000890.mp4
Resolution: 1280x720 @ 24fps
Duration: 24.4 seconds

Events Processed: 4
Intrusions Detected: 2
  - Event 1: 1 person (35.7% confidence)  
  - Event 2: 3 people (56.9%, 42.6%, 31.5% confidence)
False Alarms Filtered: 2
  - Motion detected but no people/vehicles
  
Intrusion Rate: 50% (half of motion events were actual intrusions)
```

#### What Worked

- ✅ MobileNet-SSD runs efficiently at 79 FPS on CPU
- ✅ Motion detection successfully filters stationary objects
- ✅ Pipeline correctly identifies moving people in surveillance video
- ✅ Video clips saved for detected intrusions
- ✅ False alarms filtered when no alert classes detected

#### Architecture Decisions

| Decision | Reasoning |
|----------|-----------|
| **Event-driven clip analysis** | Eliminates need for real-time FPS, allows accurate models on CPU |
| **Recommend YOLOv8s** | Best accuracy for clip analysis, 2.6s per event is acceptable |
| Support all YOLOv8 variants | Frontend can offer model selection based on accuracy/speed preference |
| Keep MobileNet as fallback | For extremely high event volumes only |
| Frame skip every 3 frames | Reduces analysis time from 7.7s to 2.6s per clip |
| Resize to 640x480 for detection | Balances speed vs. small object detection |
| Sequential event queue | Simple, reliable, no race conditions |
| Motion-gated object detection | Only run inference when motion detected |

#### Next Steps

- [ ] Test with more diverse video samples (night, weather, vehicles)
- [ ] Measure memory usage with multiple simultaneous cameras
- [ ] Implement real RTSP stream integration
- [ ] Add detection confidence threshold tuning
- [ ] Create frontend for model selection

---

## Technical Notes from Industry Partner

- Motion events sent via TCP from existing BIL Security software
- Cameras must be ONVIF Profile S or T compliant
  - [Profile S](https://www.onvif.org/profiles/profile-s/) - Streaming
  - [Profile T](https://www.onvif.org/profiles/profile-t/) - Advanced streaming
- Target hardware: Windows PC, i5/i7 processor, ~4GB RAM
- Goal: Support 10 simultaneous camera feeds

---

## References

- [OpenCV VideoCapture RTSP](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [ONVIF Profiles](https://www.onvif.org/profiles/)
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)

---

*Last updated: January 25, 2026*
