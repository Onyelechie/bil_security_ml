"""
Test Panel - Test detection with videos or webcam.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QLabel, QFileDialog, QGroupBox,
    QFormLayout, QSpinBox, QCheckBox, QProgressBar,
    QTextEdit, QSplitter, QFrame, QSlider, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QImage, QPixmap

from ..detect.config import AVAILABLE_MODELS, DEFAULT_MODEL, MODEL_INFO
from ..detect.detector import create_detector
from ..detect.motion import MotionDetector
from .zone_editor import Zone, ZoneEditorDialog


class VideoTestThread(QThread):
    """Thread for running video detection without blocking UI."""
    
    frame_ready = Signal(np.ndarray, list)  # frame, detections
    progress_update = Signal(int, int)  # current, total
    detection_result = Signal(dict)  # detection summary
    finished_processing = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, source, model_name, confidence, use_motion_filter, zones=None):
        super().__init__()
        self.source = source
        self.model_name = model_name
        self.confidence = confidence
        self.use_motion_filter = use_motion_filter
        self.zones = zones or []  # List of Zone objects
        self._running = True
        self._paused = False
        
    def run(self):
        """Process video/webcam and emit frames with detections."""
        try:
            # Open video source
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # DirectShow for Windows webcam
            else:
                cap = cv2.VideoCapture(self.source)
                
            if not cap.isOpened():
                self.error_occurred.emit(f"Could not open video source: {self.source}")
                return
                
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Create detector
            detector = create_detector(self.model_name)
            motion_detector = MotionDetector() if self.use_motion_filter else None
            
            # Stats
            frame_count = 0
            detection_count = 0
            intrusion_count = 0
            objects_detected = {}
            
            while self._running:
                if self._paused:
                    self.msleep(100)
                    continue
                    
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.source, int):
                        # Webcam - keep trying
                        continue
                    else:
                        # Video file ended
                        break
                        
                frame_count += 1
                
                # Run detection - returns list of Detection objects
                raw_detections = detector.detect(frame, self.confidence)
                
                # Convert Detection objects to dicts for easier handling
                detections = []
                for det in raw_detections:
                    detections.append({
                        'box': det.bbox,
                        'label': det.class_name,
                        'confidence': det.confidence,
                        'has_motion': True,  # Default, will be updated below
                        'in_zone': True  # Default, will be updated below
                    })
                
                # Filter by zones if defined
                if self.zones:
                    for det in detections:
                        bbox = det['box']
                        det['in_zone'] = any(z.contains_box(bbox) for z in self.zones if z.enabled)
                
                # Filter by motion if enabled
                if motion_detector and detections:
                    has_motion, motion_regions, motion_mask = motion_detector.detect(frame)
                    if motion_mask is not None:
                        for det in detections:
                            x1, y1, x2, y2 = det['box']
                            roi = motion_mask[y1:y2, x1:x2]
                            if roi.size > 0 and np.mean(roi) > 5:
                                det['has_motion'] = True
                            else:
                                det['has_motion'] = False
                
                # Count intrusions (detections with motion AND in zone)
                for det in detections:
                    if det['has_motion'] and det['in_zone']:
                        intrusion_count += 1
                
                # Count detections
                detection_count += len(detections)
                for det in detections:
                    label = det['label']
                    objects_detected[label] = objects_detected.get(label, 0) + 1
                
                # Emit frame with detections (include zones for drawing)
                self.frame_ready.emit(frame.copy(), detections)
                
                # Emit progress
                if total_frames > 0:
                    self.progress_update.emit(frame_count, total_frames)
                    
                # Control playback speed (roughly real-time for video files)
                if not isinstance(self.source, int):
                    self.msleep(int(1000 / fps / 2))  # Half speed for processing time
                    
            cap.release()
            
            # Emit final results
            self.detection_result.emit({
                'frames': frame_count,
                'detections': detection_count,
                'intrusions': intrusion_count if self.use_motion_filter else detection_count,
                'objects': objects_detected,
            })
            
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished_processing.emit()
            
    def pause(self):
        """Pause processing."""
        self._paused = True
        
    def resume(self):
        """Resume processing."""
        self._paused = False
        
    def stop(self):
        """Stop processing."""
        self._running = False
        self._paused = False


class TestPanel(QWidget):
    """Panel for testing detection with videos or webcam."""
    
    # Signals
    test_started = Signal()
    test_stopped = Signal()
    
    def __init__(self):
        super().__init__()
        self.test_thread = None
        self.current_frame = None
        self.zones = []  # List of Zone objects for test
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for video and controls
        splitter = QSplitter(Qt.Vertical)
        
        # Top: Video display
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.StyledPanel)
        video_layout = QVBoxLayout(video_frame)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a; color: white;")
        self.video_label.setText("No video loaded\n\nSelect a video file or webcam to test detection")
        video_layout.addWidget(self.video_label)
        
        # Video controls
        video_controls = QHBoxLayout()
        
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_playback)
        video_controls.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton("⬛ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_test)
        video_controls.addWidget(self.btn_stop)
        
        video_controls.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        video_controls.addWidget(self.progress_bar)
        
        self.fps_label = QLabel("FPS: --")
        video_controls.addWidget(self.fps_label)
        
        video_layout.addLayout(video_controls)
        
        splitter.addWidget(video_frame)
        
        # Bottom: Controls and log
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        # Left: Source and model settings
        settings_group = QGroupBox("Test Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Source selection
        source_layout = QHBoxLayout()
        self.combo_source = QComboBox()
        self.combo_source.addItems(["Video File", "Webcam (0)", "Webcam (1)"])
        source_layout.addWidget(self.combo_source)
        
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_video)
        source_layout.addWidget(self.btn_browse)
        settings_layout.addRow("Source:", source_layout)
        
        self.label_file = QLabel("No file selected")
        self.label_file.setStyleSheet("color: gray; font-style: italic;")
        settings_layout.addRow("", self.label_file)
        
        # Model selection
        self.combo_model = QComboBox()
        for model_key in AVAILABLE_MODELS:
            info = MODEL_INFO.get(model_key, {})
            display = f"{model_key} ({info.get('speed', 'N/A')})"
            self.combo_model.addItem(display, model_key)
        # Set default
        for i in range(self.combo_model.count()):
            if self.combo_model.itemData(i) == DEFAULT_MODEL:
                self.combo_model.setCurrentIndex(i)
                break
        settings_layout.addRow("Model:", self.combo_model)
        
        # Confidence threshold
        self.spin_confidence = QSpinBox()
        self.spin_confidence.setRange(10, 100)
        self.spin_confidence.setValue(50)
        self.spin_confidence.setSuffix("%")
        settings_layout.addRow("Confidence:", self.spin_confidence)
        
        # Motion filter
        self.check_motion = QCheckBox("Filter by motion")
        self.check_motion.setChecked(True)
        settings_layout.addRow("", self.check_motion)
        
        # Zone filter
        self.check_zones = QCheckBox("Filter by zones")
        self.check_zones.setChecked(False)
        self.check_zones.stateChanged.connect(self._on_zone_filter_changed)
        settings_layout.addRow("", self.check_zones)
        
        # Zone editor button
        zone_layout = QHBoxLayout()
        self.btn_edit_zones = QPushButton("Edit Zones")
        self.btn_edit_zones.setEnabled(False)
        self.btn_edit_zones.clicked.connect(self._edit_zones)
        zone_layout.addWidget(self.btn_edit_zones)
        
        self.label_zones = QLabel("No zones defined")
        self.label_zones.setStyleSheet("color: gray; font-style: italic;")
        zone_layout.addWidget(self.label_zones)
        zone_layout.addStretch()
        settings_layout.addRow("", zone_layout)
        
        # Alert object types - what triggers alerts
        alert_objects_layout = QVBoxLayout()
        alert_objects_label = QLabel("Alert on:")
        alert_objects_layout.addWidget(alert_objects_label)
        
        alert_checks_layout = QHBoxLayout()
        self.check_alert_person = QCheckBox("Person")
        self.check_alert_person.setChecked(True)
        alert_checks_layout.addWidget(self.check_alert_person)
        
        self.check_alert_vehicle = QCheckBox("Vehicle")
        self.check_alert_vehicle.setChecked(True)
        alert_checks_layout.addWidget(self.check_alert_vehicle)
        
        self.check_alert_other = QCheckBox("Other")
        self.check_alert_other.setChecked(False)
        alert_checks_layout.addWidget(self.check_alert_other)
        alert_objects_layout.addLayout(alert_checks_layout)
        
        settings_layout.addRow("", alert_objects_layout)
        
        # Start button
        self.btn_start_test = QPushButton("Start Test")
        self.btn_start_test.setStyleSheet("font-weight: bold; padding: 10px;")
        self.btn_start_test.clicked.connect(self._start_test)
        settings_layout.addRow("", self.btn_start_test)
        
        bottom_layout.addWidget(settings_group)
        
        # Right: Tabbed Results/Alerts
        from PySide6.QtWidgets import QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
        
        right_tabs = QTabWidget()
        
        # === Results Tab ===
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(5, 5, 5, 5)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # Stats labels
        stats_layout = QHBoxLayout()
        
        self.label_frames = QLabel("Frames: 0")
        stats_layout.addWidget(self.label_frames)
        
        self.label_detections = QLabel("Detections: 0")
        stats_layout.addWidget(self.label_detections)
        
        self.label_intrusions = QLabel("Intrusions: 0")
        self.label_intrusions.setStyleSheet("color: red; font-weight: bold;")
        stats_layout.addWidget(self.label_intrusions)
        
        stats_layout.addStretch()
        results_layout.addLayout(stats_layout)
        
        right_tabs.addTab(results_widget, "Results")
        
        # === Alerts Tab ===
        alerts_widget = QWidget()
        alerts_layout = QVBoxLayout(alerts_widget)
        alerts_layout.setContentsMargins(5, 5, 5, 5)
        
        # Alert controls - top row
        alert_controls = QHBoxLayout()
        
        self.label_alert_count = QLabel("Alerts: 0")
        self.label_alert_count.setStyleSheet("font-weight: bold; color: red;")
        alert_controls.addWidget(self.label_alert_count)
        
        self.label_filtered_count = QLabel("(showing all)")
        self.label_filtered_count.setStyleSheet("color: gray; font-style: italic;")
        alert_controls.addWidget(self.label_filtered_count)
        
        alert_controls.addStretch()
        
        self.btn_clear_alerts = QPushButton("Clear")
        self.btn_clear_alerts.clicked.connect(self._clear_alerts)
        alert_controls.addWidget(self.btn_clear_alerts)
        
        self.btn_export_alerts = QPushButton("Export")
        self.btn_export_alerts.clicked.connect(self._export_alerts)
        alert_controls.addWidget(self.btn_export_alerts)
        
        alerts_layout.addLayout(alert_controls)
        
        # Alert filter controls - second row
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filter:"))
        
        # Object type filter
        filter_layout.addWidget(QLabel("Object:"))
        self.combo_alert_object = QComboBox()
        self.combo_alert_object.addItem("All Objects", "all")
        self.combo_alert_object.addItem("Person", "person")
        self.combo_alert_object.addItem("Vehicles", "vehicle")
        self.combo_alert_object.addItem("Car", "car")
        self.combo_alert_object.addItem("Truck", "truck")
        self.combo_alert_object.addItem("Motorcycle", "motorcycle")
        self.combo_alert_object.addItem("Bicycle", "bicycle")
        self.combo_alert_object.currentIndexChanged.connect(self._apply_alert_filter)
        filter_layout.addWidget(self.combo_alert_object)
        
        # Zone filter
        filter_layout.addWidget(QLabel("Zone:"))
        self.combo_alert_zone = QComboBox()
        self.combo_alert_zone.addItem("All Zones", "all")
        self.combo_alert_zone.currentIndexChanged.connect(self._apply_alert_filter)
        filter_layout.addWidget(self.combo_alert_zone)
        
        filter_layout.addStretch()
        alerts_layout.addLayout(filter_layout)
        
        # Alert table
        self.alert_table = QTableWidget()
        self.alert_table.setColumnCount(5)
        self.alert_table.setHorizontalHeaderLabels([
            "Time", "Frame", "Object", "Confidence", "Zone"
        ])
        header = self.alert_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.alert_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.alert_table.setAlternatingRowColors(True)
        alerts_layout.addWidget(self.alert_table)
        
        right_tabs.addTab(alerts_widget, "Alerts")
        
        bottom_layout.addWidget(right_tabs)
        
        splitter.addWidget(bottom_widget)
        
        # Set splitter sizes
        splitter.setSizes([500, 200])
        
        layout.addWidget(splitter)
        
        # Video file path storage
        self.video_path = None
        
        # Alert storage
        self.alerts = []
        self.alert_count = 0
        self.visible_alert_count = 0
        
        # FPS calculation
        self.frame_times = []
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._update_fps)
        
    def _browse_video(self):
        """Browse for video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*.*)"
        )
        if file_path:
            self.video_path = file_path
            self.label_file.setText(Path(file_path).name)
            self.combo_source.setCurrentIndex(0)  # Set to "Video File"
            self._log(f"Selected: {file_path}")
            # Enable zone editing once we have a video
            self.btn_edit_zones.setEnabled(True)
            # Clear old zones when new video selected
            self.zones = []
            self._update_zone_label()
            
    def _on_zone_filter_changed(self, state):
        """Handle zone filter checkbox change."""
        if state == Qt.Checked and not self.zones:
            self._log("No zones defined. Click 'Edit Zones' to define monitoring areas.")
            
    def _edit_zones(self):
        """Open zone editor dialog."""
        source = self._get_source()
        if source is None:
            QMessageBox.warning(self, "No Source", "Please select a video file first.")
            return
            
        # Get first frame from video for background
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Could not open video source.")
            return
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.warning(self, "Error", "Could not read frame from video.")
            return
            
        # Open zone editor dialog
        dialog = ZoneEditorDialog(
            self,
            camera_id="test",
            initial_frame=frame,
            existing_zones=self.zones
        )
        
        if dialog.exec():
            self.zones = dialog.get_zones()
            self._update_zone_label()
            self._log(f"Zones updated: {len(self.zones)} zone(s) defined")
            if self.zones:
                self.check_zones.setChecked(True)
                
    def _update_zone_label(self):
        """Update the zone count label."""
        if self.zones:
            enabled = sum(1 for z in self.zones if z.enabled)
            self.label_zones.setText(f"{len(self.zones)} zone(s) ({enabled} enabled)")
            self.label_zones.setStyleSheet("color: green;")
        else:
            self.label_zones.setText("No zones defined")
            self.label_zones.setStyleSheet("color: gray; font-style: italic;")
        
        # Update zone filter combo in alerts tab
        self._update_alert_zone_filter()
            
    def _get_source(self):
        """Get the video source based on selection."""
        source_text = self.combo_source.currentText()
        if source_text == "Video File":
            return self.video_path
        elif "Webcam (0)" in source_text:
            return 0
        elif "Webcam (1)" in source_text:
            return 1
        return None
        
    def _start_test(self):
        """Start the test."""
        source = self._get_source()
        
        if source is None:
            self._log("ERROR: Please select a video file or webcam")
            return
            
        if self.test_thread and self.test_thread.isRunning():
            self._log("Test already running")
            return
            
        model = self.combo_model.currentData()
        confidence = self.spin_confidence.value() / 100.0
        use_motion = self.check_motion.isChecked()
        use_zones = self.check_zones.isChecked()
        
        self._log(f"Starting test with {model} (confidence: {confidence:.0%})")
        self._log(f"Motion filter: {'enabled' if use_motion else 'disabled'}")
        if use_zones and self.zones:
            enabled_zones = [z for z in self.zones if z.enabled]
            self._log(f"Zone filter: {len(enabled_zones)} zone(s) active")
        else:
            self._log("Zone filter: disabled (monitoring entire frame)")
        
        # Reset stats
        self.label_frames.setText("Frames: 0")
        self.label_detections.setText("Detections: 0")
        self.label_intrusions.setText("Intrusions: 0")
        self.progress_bar.setValue(0)
        self.frame_times = []
        
        # Reset alerts
        self.alerts = []
        self.alert_count = 0
        self.visible_alert_count = 0
        self.alert_table.setRowCount(0)
        self.label_alert_count.setText("Alerts: 0")
        self.label_filtered_count.setText("(showing all)")
        self.current_frame_number = 0
        
        # Reset alert filters
        self.combo_alert_object.setCurrentIndex(0)  # All Objects
        self.combo_alert_zone.setCurrentIndex(0)    # All Zones
        self._update_alert_zone_filter()
        
        # Get zones to use
        zones_to_use = self.zones if use_zones else []
        
        # Create and start thread
        self.test_thread = VideoTestThread(source, model, confidence, use_motion, zones_to_use)
        self.test_thread.frame_ready.connect(self._on_frame_ready)
        self.test_thread.progress_update.connect(self._on_progress)
        self.test_thread.detection_result.connect(self._on_result)
        self.test_thread.finished_processing.connect(self._on_finished)
        self.test_thread.error_occurred.connect(self._on_error)
        
        self.test_thread.start()
        
        # Update UI
        self.btn_start_test.setEnabled(False)
        self.btn_play.setEnabled(True)
        self.btn_play.setText("⏸ Pause")
        self.btn_stop.setEnabled(True)
        self.fps_timer.start(500)
        
        self.test_started.emit()
        
    def _stop_test(self):
        """Stop the test."""
        if self.test_thread:
            self.test_thread.stop()
            self.test_thread.wait(2000)
            self.test_thread = None
            
        self._on_finished()
        
    def _toggle_playback(self):
        """Toggle pause/resume."""
        if not self.test_thread:
            return
            
        if self.test_thread._paused:
            self.test_thread.resume()
            self.btn_play.setText("⏸ Pause")
        else:
            self.test_thread.pause()
            self.btn_play.setText("▶ Resume")
            
    @Slot(np.ndarray, list)
    def _on_frame_ready(self, frame, detections):
        """Handle new frame with detections."""
        self.frame_times.append(datetime.now())
        
        # Draw zones on frame first (so detections draw on top)
        use_zones = self.check_zones.isChecked()
        if use_zones and self.zones:
            for zone in self.zones:
                if zone.enabled and len(zone.points) >= 3:
                    pts = np.array(zone.points, dtype=np.int32)
                    # Draw filled semi-transparent zone
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], zone.color)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                    # Draw zone border
                    cv2.polylines(frame, [pts], True, zone.color, 2)
                    # Draw zone name
                    cx = int(np.mean([p[0] for p in zone.points]))
                    cy = int(np.mean([p[1] for p in zone.points]))
                    cv2.putText(frame, zone.name, (cx - 30, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw detections on frame
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['confidence']
            has_motion = det.get('has_motion', True)
            in_zone = det.get('in_zone', True)
            
            # Color based on motion and zone status
            # Red = intrusion (motion + in zone)
            # Yellow = no motion (filtered)
            # Gray = outside zone (filtered)
            if not in_zone:
                color = (128, 128, 128)  # Gray for outside zone
            elif has_motion:
                color = (0, 0, 255)  # Red for intrusion
            else:
                color = (0, 255, 255)  # Yellow for no motion
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label_text = f"{label}: {conf:.0%}"
            if not in_zone:
                label_text += " (outside zone)"
            elif not has_motion:
                label_text += " (no motion)"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add to alerts if it's an intrusion (passes all enabled filters)
            # Check zone filter - only add alert if in zone OR zone filtering is disabled
            zone_pass = in_zone  # Already set correctly based on check_zones setting
            
            # Check motion filter
            motion_pass = has_motion or not self.check_motion.isChecked()
            
            # Check object type filter
            object_pass = self._should_alert_on_object(label)
            
            # Check confidence threshold
            min_confidence = self.spin_confidence.value() / 100.0
            confidence_pass = conf >= min_confidence
            
            if zone_pass and motion_pass and object_pass and confidence_pass:
                self._add_alert(det, self.current_frame_number)
                       
        # Convert to QPixmap and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)
        
        self.current_frame = frame
        
    @Slot(int, int)
    def _on_progress(self, current, total):
        """Update progress bar."""
        self.current_frame_number = current
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.label_frames.setText(f"Frames: {current}/{total}")
        else:
            self.label_frames.setText(f"Frames: {current}")
            
    @Slot(dict)
    def _on_result(self, result):
        """Handle final detection results."""
        self._log("\n=== TEST COMPLETE ===")
        self._log(f"Total frames: {result['frames']}")
        self._log(f"Total detections: {result['detections']}")
        self._log(f"Intrusions (in zone + motion): {result['intrusions']}")
        
        if result['objects']:
            self._log("\nObjects detected:")
            for obj, count in sorted(result['objects'].items(), key=lambda x: -x[1]):
                self._log(f"  - {obj}: {count}")
                
        self.label_detections.setText(f"Detections: {result['detections']}")
        self.label_intrusions.setText(f"Intrusions: {result['intrusions']}")
        
    @Slot()
    def _on_finished(self):
        """Handle test finished."""
        self.btn_start_test.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.btn_play.setText("▶ Play")
        self.btn_stop.setEnabled(False)
        self.fps_timer.stop()
        
        self._log("Test stopped")
        self.test_stopped.emit()
        
    @Slot(str)
    def _on_error(self, error):
        """Handle error."""
        self._log(f"ERROR: {error}")
        
    def _update_fps(self):
        """Calculate and display FPS."""
        now = datetime.now()
        # Keep only recent frame times (last 2 seconds)
        self.frame_times = [t for t in self.frame_times 
                          if (now - t).total_seconds() < 2]
        
        if len(self.frame_times) >= 2:
            duration = (self.frame_times[-1] - self.frame_times[0]).total_seconds()
            if duration > 0:
                fps = (len(self.frame_times) - 1) / duration
                self.fps_label.setText(f"FPS: {fps:.1f}")
            else:
                self.fps_label.setText("FPS: --")
        else:
            self.fps_label.setText("FPS: --")
            
    def _log(self, message):
        """Add message to results log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def _add_alert(self, detection: dict, frame_number: int):
        """Add an alert to the alerts table."""
        from PySide6.QtWidgets import QTableWidgetItem
        from PySide6.QtGui import QColor
        
        timestamp = datetime.now()
        
        # Store alert
        alert = {
            'timestamp': timestamp,
            'frame': frame_number,
            'label': detection['label'],
            'confidence': detection['confidence'],
            'box': detection['box'],
            'zone': self._get_zone_for_detection(detection) if self.zones else "Entire Frame"
        }
        self.alerts.append(alert)
        self.alert_count += 1
        
        # Add row to table
        row = self.alert_table.rowCount()
        self.alert_table.insertRow(row)
        
        # Time
        time_item = QTableWidgetItem(timestamp.strftime("%H:%M:%S.%f")[:-3])
        self.alert_table.setItem(row, 0, time_item)
        
        # Frame
        frame_item = QTableWidgetItem(str(frame_number))
        self.alert_table.setItem(row, 1, frame_item)
        
        # Object
        object_item = QTableWidgetItem(detection['label'])
        # Color code by object type
        if detection['label'] == 'person':
            object_item.setBackground(QColor(255, 200, 200))  # Light red
        elif detection['label'] in ['car', 'truck', 'bus', 'motorcycle']:
            object_item.setBackground(QColor(200, 200, 255))  # Light blue
        self.alert_table.setItem(row, 2, object_item)
        
        # Confidence
        conf_item = QTableWidgetItem(f"{detection['confidence']:.0%}")
        self.alert_table.setItem(row, 3, conf_item)
        
        # Zone
        zone_name = alert['zone']
        zone_item = QTableWidgetItem(zone_name)
        self.alert_table.setItem(row, 4, zone_item)
        
        # Update alert count label
        self.label_alert_count.setText(f"Alerts: {self.alert_count}")
        
        # Scroll to bottom
        self.alert_table.scrollToBottom()
        
    def _get_zone_for_detection(self, detection: dict) -> str:
        """Get the zone name that contains this detection."""
        if not self.zones:
            return "Entire Frame"
            
        bbox = detection['box']
        for zone in self.zones:
            if zone.enabled and zone.contains_box(bbox):
                return zone.name
        return "Unknown"
    
    def _should_alert_on_object(self, label: str) -> bool:
        """Check if we should generate an alert for this object type."""
        label_lower = label.lower()
        
        # Person check
        if label_lower == 'person':
            return self.check_alert_person.isChecked()
        
        # Vehicle check (car, truck, bus, motorcycle, bicycle)
        vehicle_types = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane', 'train'}
        if label_lower in vehicle_types:
            return self.check_alert_vehicle.isChecked()
        
        # Other objects (animals, items, etc.)
        return self.check_alert_other.isChecked()
        
    def _clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
        self.alert_count = 0
        self.visible_alert_count = 0
        self.alert_table.setRowCount(0)
        self.label_alert_count.setText("Alerts: 0")
        self.label_filtered_count.setText("(showing all)")
        self._log("Alerts cleared")
        
    def _update_alert_zone_filter(self):
        """Update the zone filter dropdown with current zones."""
        # Remember current selection
        current_data = self.combo_alert_zone.currentData()
        
        # Block signals to prevent triggering filter during update
        self.combo_alert_zone.blockSignals(True)
        
        # Clear and rebuild
        self.combo_alert_zone.clear()
        self.combo_alert_zone.addItem("All Zones", "all")
        
        # Add each zone
        if self.zones:
            for zone in self.zones:
                self.combo_alert_zone.addItem(zone.name, zone.name)
        
        # Also add "Entire Frame" for when no zones are defined
        self.combo_alert_zone.addItem("Entire Frame", "Entire Frame")
        
        # Restore selection if possible
        idx = self.combo_alert_zone.findData(current_data)
        if idx >= 0:
            self.combo_alert_zone.setCurrentIndex(idx)
        
        self.combo_alert_zone.blockSignals(False)
        
    def _apply_alert_filter(self):
        """Apply filters to the alerts table."""
        object_filter = self.combo_alert_object.currentData()
        zone_filter = self.combo_alert_zone.currentData()
        
        # Vehicle types for grouping
        vehicle_types = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
        
        visible_count = 0
        
        for row in range(self.alert_table.rowCount()):
            # Get the object and zone from this row
            object_item = self.alert_table.item(row, 2)  # Object column
            zone_item = self.alert_table.item(row, 4)    # Zone column
            
            if object_item is None or zone_item is None:
                continue
                
            obj_label = object_item.text().lower()
            zone_name = zone_item.text()
            
            # Check object filter
            object_match = True
            if object_filter != "all":
                if object_filter == "vehicle":
                    object_match = obj_label in vehicle_types
                else:
                    object_match = (obj_label == object_filter)
            
            # Check zone filter
            zone_match = True
            if zone_filter != "all":
                zone_match = (zone_name == zone_filter)
            
            # Show/hide row
            show_row = object_match and zone_match
            self.alert_table.setRowHidden(row, not show_row)
            
            if show_row:
                visible_count += 1
        
        # Update filtered count label
        self.visible_alert_count = visible_count
        total = self.alert_count
        
        if object_filter == "all" and zone_filter == "all":
            self.label_filtered_count.setText("(showing all)")
        else:
            self.label_filtered_count.setText(f"(showing {visible_count}/{total})")
        
    def _export_alerts(self):
        """Export alerts to CSV file."""
        if not self.alerts:
            QMessageBox.information(self, "No Alerts", "No alerts to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Alerts",
            f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Frame', 'Object', 'Confidence', 'Zone', 'BBox'])
                    for alert in self.alerts:
                        writer.writerow([
                            alert['timestamp'].isoformat(),
                            alert['frame'],
                            alert['label'],
                            f"{alert['confidence']:.2%}",
                            alert['zone'],
                            str(alert['box'])
                        ])
                self._log(f"Exported {len(self.alerts)} alerts to {file_path}")
                QMessageBox.information(self, "Export Complete", 
                                       f"Exported {len(self.alerts)} alerts to:\n{file_path}")
            except Exception as e:
                self._log(f"Export failed: {e}")
                QMessageBox.warning(self, "Export Failed", f"Could not export alerts:\n{e}")
        
    def closeEvent(self, event):
        """Clean up when closing."""
        self._stop_test()
        event.accept()
