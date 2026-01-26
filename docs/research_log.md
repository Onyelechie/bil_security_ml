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

- TCP listener successfully receives and parses JSON events
- Ring buffer correctly stores and retrieves timestamped frames
- Video clip saving with OpenCV VideoWriter (mp4v codec)
- End-to-end pipeline: event -> frame extraction -> MP4 output

#### What Failed / Challenges

- Need to handle case where event arrives before buffer is full (< 2s of history)

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
| RTSP decode CPU cost | Decoding RTSP will be a significant CPU bottleneck | Resolution, codec, FPS | CPU %, decode latency (ms) | Waiting for RTSP URL |
| FPS vs accuracy tradeoff | Lower FPS reduces CPU but may miss fast motion | 5, 10, 15, 30 FPS | CPU %, detection recall, false negatives | Pending |
| Frame window size impact | Larger window improves context but increases processing | +/-2s, +/-5s, +/-10s | Detection accuracy, clip size, processing time | Pending |
| Lightweight model comparison | MobileNet/YOLO-Nano vs full models | Model architecture | mAP, inference time, RAM usage | COMPLETED |
| Multi-stream scaling | 10 simultaneous streams on i5/i7 | Number of streams | CPU %, memory, dropped frames | Pending |
| Motion filtering approaches | Pre-filtering reduces unnecessary inference | Background subtraction, frame diff | True positive rate, CPU savings | IMPLEMENTED |

---

### January 25, 2026 - Model Speed Benchmarks & Detection Pipeline

**Objective:** Benchmark detection model speeds and complete the intrusion detection pipeline.

#### Model Speed Benchmark Results

**Test Setup:**
- Test Video: VIRAT_S_010204_05_000856_000890.mp4 (1280x720, VIRAT surveillance dataset)
- System: Windows PC (CPU-only inference, no GPU)

##### Benchmark at Native Resolution (1280x720) - UPDATED

| Model | FPS | Avg (ms) | Detections | Status |
|-------|-----|----------|------------|--------|
| MobileNet-SSD | 15.5 | 64.7 | 5 per frame | Slower than expected at HD |
| **YOLOv8n** | **20.4** | **48.9** | 7 per frame | **RECOMMENDED - Best speed/accuracy** |
| YOLOv8s | 13.7 | 73.1 | 9 per frame | Best accuracy, acceptable speed |

##### Benchmark at Resized Resolution (640x480) - Previous Test

| Model | FPS | Avg (ms) | Classes | Notes |
|-------|-----|----------|---------|-------|
| MobileNet-SSD | 90.4 | 11.1 | 21 | Fast only at low resolution |
| YOLOv8n | 29.0 | 34.5 | 80 | Consistent performance |
| YOLOv8s | 14.0 | 71.3 | 80 | Consistent performance |

#### IMPORTANT FINDING: MobileNet Resolution Scaling Issue

**MobileNet-SSD does NOT scale well to HD resolution:**

| Resolution | MobileNet FPS | YOLOv8n FPS | Winner |
|------------|---------------|-------------|--------|
| 640×480 | 90.4 | 29.0 | MobileNet (3x faster) |
| **1280×720** | **15.5** | **20.4** | **YOLOv8n (31% faster)** |

**Why this matters:**
- Surveillance cameras typically output 720p or 1080p
- MobileNet must resize frames internally, losing detail
- YOLOv8 handles native resolution better
- **MobileNet is NOT recommended for HD surveillance video**

#### Updated Key Findings

1. **YOLOv8n is now the PRIMARY recommendation**
   - 20.4 FPS at native 720p resolution - faster than MobileNet!
   - Better detection accuracy (7 detections vs 5 for MobileNet)
   - 80 classes vs MobileNet's 21
   - Best for: ALL camera types (close and distant)

2. **YOLOv8s for maximum accuracy**
   - 13.7 FPS at 720p - still acceptable for event-driven analysis
   - 9 detections per frame (best accuracy)
   - Best for: Wide areas where small/distant objects matter

3. **MobileNet-SSD DEMOTED to fallback only**
   - ~~79 FPS~~ → Only 15.5 FPS at real surveillance resolution
   - Fewer detections (5 vs 7-9 for YOLO)
   - Only 21 classes (missing many vehicle types)
   - Use only for: Extremely low-powered hardware or pre-resized 480p streams

4. **Event-Driven vs Real-Time Streaming: Why YOLO Works on CPU**

   Our architecture analyzes **short clips after motion events**, not continuous live streams.
   This fundamentally changes the hardware requirements:

   | Architecture | Description | FPS Requirement | YOLOv8s Viable? |
   |--------------|-------------|-----------------|------------------|
   | **Real-Time Streaming** | Process every frame from all cameras continuously | 150 FPS (10 cams x 15fps) | No (only 13.6 FPS) |
   | **Event-Driven Clips** | Analyze 7-second clips when motion detected | ~35 frames per event | Yes |

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
| CPU only (i5/i7) | ~2 FPS | 0 cameras | Works fine |
| NVIDIA RTX 3060 | ~80-120 FPS | 5-8 cameras | Overkill |
| NVIDIA RTX 3080 | ~150-200 FPS | 10+ cameras | Overkill |

#### Model Recommendations by Camera Count

**For Event-Driven Clip Analysis (BIL Security Architecture):**

| Cameras | Recommended Model | Analysis Time/Clip | Events/Minute Capacity | Notes |
|---------|-------------------|-------------------|------------------------|-------|
| 1-10 | **YOLOv8n** | ~1.7 sec | ~35 events | Best speed + accuracy at 720p (RECOMMENDED) |
| 1-10 (max accuracy) | YOLOv8s | ~2.6 sec | ~23 events | More detections, slower |
| Legacy/Low-power | MobileNet | ~2.3 sec | ~26 events | Only if YOLO won't run |

**Model Selection Guide:**

| Scenario | Model | Why |
|----------|-------|-----|
| **Default choice** | YOLOv8n | Fastest at 720p, good accuracy |
| Maximum accuracy needed | YOLOv8s | More detections, 30% slower |
| Very old hardware | MobileNet | Lighter model, but worse at HD |
| 480p cameras only | MobileNet | Fast at low resolution |

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

- MobileNet-SSD runs efficiently at 79 FPS on CPU (at 480p)
- Motion detection successfully filters stationary objects
- Pipeline correctly identifies moving people in surveillance video
- Video clips saved for detected intrusions
- False alarms filtered when no alert classes detected

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
