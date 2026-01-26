"""
Settings Dialog - Configure application settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QFormLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QLineEdit, QCheckBox, QGroupBox,
    QDialogButtonBox, QLabel, QPushButton, QFileDialog,
    QMessageBox
)
from PySide6.QtCore import Qt

from ..detect.config import (
    AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_CONFIDENCE,
    DEFAULT_MOTION_THRESHOLD, MODEL_INFO
)


class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 400)
        
        self._setup_ui()
        self._load_settings()
        
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Tab widget
        tabs = QTabWidget()
        
        # Detection tab
        detection_tab = self._create_detection_tab()
        tabs.addTab(detection_tab, "Detection")
        
        # Network tab
        network_tab = self._create_network_tab()
        tabs.addTab(network_tab, "Network")
        
        # Storage tab
        storage_tab = self._create_storage_tab()
        tabs.addTab(storage_tab, "Storage")
        
        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced")
        
        layout.addWidget(tabs)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply_settings)
        layout.addWidget(buttons)
        
    def _create_detection_tab(self):
        """Create detection settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model settings
        model_group = QGroupBox("Detection Model")
        model_layout = QFormLayout(model_group)
        
        self.combo_default_model = QComboBox()
        for model_key, info in MODEL_INFO.items():
            display = f"{model_key} - {info['description']}"
            self.combo_default_model.addItem(display, model_key)
        self.combo_default_model.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addRow("Default Model:", self.combo_default_model)
        
        self.label_model_info = QLabel()
        self.label_model_info.setWordWrap(True)
        self.label_model_info.setStyleSheet("color: gray; font-style: italic;")
        model_layout.addRow("", self.label_model_info)
        
        layout.addWidget(model_group)
        
        # Threshold settings
        threshold_group = QGroupBox("Detection Thresholds")
        threshold_layout = QFormLayout(threshold_group)
        
        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0.1, 1.0)
        self.spin_confidence.setSingleStep(0.05)
        self.spin_confidence.setDecimals(2)
        self.spin_confidence.setValue(DEFAULT_CONFIDENCE)
        threshold_layout.addRow("Confidence Threshold:", self.spin_confidence)
        
        self.spin_motion = QDoubleSpinBox()
        self.spin_motion.setRange(0.001, 0.1)
        self.spin_motion.setSingleStep(0.005)
        self.spin_motion.setDecimals(3)
        self.spin_motion.setValue(DEFAULT_MOTION_THRESHOLD)
        threshold_layout.addRow("Motion Threshold:", self.spin_motion)
        
        layout.addWidget(threshold_group)
        
        # False alarm filtering
        filter_group = QGroupBox("False Alarm Filtering")
        filter_layout = QFormLayout(filter_group)
        
        self.check_motion_filter = QCheckBox("Enable motion-based filtering")
        self.check_motion_filter.setChecked(True)
        filter_layout.addRow(self.check_motion_filter)
        
        self.spin_min_detections = QSpinBox()
        self.spin_min_detections.setRange(1, 10)
        self.spin_min_detections.setValue(2)
        filter_layout.addRow("Min detections to confirm:", self.spin_min_detections)
        
        layout.addWidget(filter_group)
        
        layout.addStretch()
        return widget
        
    def _create_network_tab(self):
        """Create network settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # TCP listener settings
        tcp_group = QGroupBox("TCP Event Listener")
        tcp_layout = QFormLayout(tcp_group)
        
        self.edit_tcp_host = QLineEdit()
        self.edit_tcp_host.setText("0.0.0.0")
        tcp_layout.addRow("Listen Address:", self.edit_tcp_host)
        
        self.spin_tcp_port = QSpinBox()
        self.spin_tcp_port.setRange(1024, 65535)
        self.spin_tcp_port.setValue(9000)
        tcp_layout.addRow("Listen Port:", self.spin_tcp_port)
        
        self.check_tcp_enabled = QCheckBox("Enable TCP listener")
        self.check_tcp_enabled.setChecked(True)
        tcp_layout.addRow(self.check_tcp_enabled)
        
        layout.addWidget(tcp_group)
        
        # RTSP settings
        rtsp_group = QGroupBox("RTSP Settings")
        rtsp_layout = QFormLayout(rtsp_group)
        
        self.spin_rtsp_timeout = QSpinBox()
        self.spin_rtsp_timeout.setRange(5, 60)
        self.spin_rtsp_timeout.setValue(10)
        self.spin_rtsp_timeout.setSuffix(" seconds")
        rtsp_layout.addRow("Connection Timeout:", self.spin_rtsp_timeout)
        
        self.spin_reconnect_delay = QSpinBox()
        self.spin_reconnect_delay.setRange(1, 30)
        self.spin_reconnect_delay.setValue(5)
        self.spin_reconnect_delay.setSuffix(" seconds")
        rtsp_layout.addRow("Reconnect Delay:", self.spin_reconnect_delay)
        
        layout.addWidget(rtsp_group)
        
        layout.addStretch()
        return widget
        
    def _create_storage_tab(self):
        """Create storage settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Clip storage
        clip_group = QGroupBox("Event Clips")
        clip_layout = QFormLayout(clip_group)
        
        # Output directory
        dir_layout = QHBoxLayout()
        self.edit_output_dir = QLineEdit()
        self.edit_output_dir.setText("event_clips")
        dir_layout.addWidget(self.edit_output_dir)
        
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(btn_browse)
        
        clip_layout.addRow("Output Directory:", dir_layout)
        
        self.check_save_clips = QCheckBox("Save event clips")
        self.check_save_clips.setChecked(True)
        clip_layout.addRow(self.check_save_clips)
        
        self.check_save_intrusions_only = QCheckBox("Save intrusions only")
        self.check_save_intrusions_only.setChecked(True)
        clip_layout.addRow(self.check_save_intrusions_only)
        
        layout.addWidget(clip_group)
        
        # Buffer settings
        buffer_group = QGroupBox("Frame Buffer")
        buffer_layout = QFormLayout(buffer_group)
        
        self.spin_buffer_duration = QSpinBox()
        self.spin_buffer_duration.setRange(10, 120)
        self.spin_buffer_duration.setValue(30)
        self.spin_buffer_duration.setSuffix(" seconds")
        buffer_layout.addRow("Buffer Duration:", self.spin_buffer_duration)
        
        self.spin_pre_event = QSpinBox()
        self.spin_pre_event.setRange(1, 10)
        self.spin_pre_event.setValue(2)
        self.spin_pre_event.setSuffix(" seconds")
        buffer_layout.addRow("Pre-Event Window:", self.spin_pre_event)
        
        self.spin_post_event = QSpinBox()
        self.spin_post_event.setRange(1, 15)
        self.spin_post_event.setValue(5)
        self.spin_post_event.setSuffix(" seconds")
        buffer_layout.addRow("Post-Event Window:", self.spin_post_event)
        
        layout.addWidget(buffer_group)
        
        layout.addStretch()
        return widget
        
    def _create_advanced_tab(self):
        """Create advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        self.spin_frame_skip = QSpinBox()
        self.spin_frame_skip.setRange(1, 10)
        self.spin_frame_skip.setValue(5)
        perf_layout.addRow("Frame Skip (analyze every N):", self.spin_frame_skip)
        
        self.spin_max_cameras = QSpinBox()
        self.spin_max_cameras.setRange(1, 20)
        self.spin_max_cameras.setValue(10)
        perf_layout.addRow("Max Cameras:", self.spin_max_cameras)
        
        self.check_gpu = QCheckBox("Use GPU if available")
        self.check_gpu.setChecked(False)
        perf_layout.addRow(self.check_gpu)
        
        layout.addWidget(perf_group)
        
        # Logging settings
        log_group = QGroupBox("Logging")
        log_layout = QFormLayout(log_group)
        
        self.combo_log_level = QComboBox()
        self.combo_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.combo_log_level.setCurrentText("INFO")
        log_layout.addRow("Log Level:", self.combo_log_level)
        
        self.check_log_file = QCheckBox("Write to log file")
        self.check_log_file.setChecked(False)
        log_layout.addRow(self.check_log_file)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        return widget
        
    def _on_model_changed(self, index):
        """Update model info when selection changes."""
        model_key = self.combo_default_model.currentData()
        if model_key in MODEL_INFO:
            info = MODEL_INFO[model_key]
            text = f"Speed: {info['speed']} | Use case: {info['use_case']}"
            self.label_model_info.setText(text)
            
    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.edit_output_dir.setText(dir_path)
            
    def _load_settings(self):
        """Load current settings."""
        # Set default model selection
        for i in range(self.combo_default_model.count()):
            if self.combo_default_model.itemData(i) == DEFAULT_MODEL:
                self.combo_default_model.setCurrentIndex(i)
                break
        self._on_model_changed(0)
        
        # TODO: Load from persistent storage
        
    def _apply_settings(self):
        """Apply settings without closing dialog."""
        # TODO: Apply settings to running system
        QMessageBox.information(self, "Settings", "Settings applied.")
        
    def _save_and_accept(self):
        """Save settings and close dialog."""
        self._apply_settings()
        self.accept()
        
    def get_settings(self):
        """Return current settings as dict."""
        return {
            'detection': {
                'default_model': self.combo_default_model.currentData(),
                'confidence': self.spin_confidence.value(),
                'motion_threshold': self.spin_motion.value(),
                'motion_filter_enabled': self.check_motion_filter.isChecked(),
                'min_detections': self.spin_min_detections.value(),
            },
            'network': {
                'tcp_host': self.edit_tcp_host.text(),
                'tcp_port': self.spin_tcp_port.value(),
                'tcp_enabled': self.check_tcp_enabled.isChecked(),
                'rtsp_timeout': self.spin_rtsp_timeout.value(),
                'reconnect_delay': self.spin_reconnect_delay.value(),
            },
            'storage': {
                'output_dir': self.edit_output_dir.text(),
                'save_clips': self.check_save_clips.isChecked(),
                'intrusions_only': self.check_save_intrusions_only.isChecked(),
                'buffer_duration': self.spin_buffer_duration.value(),
                'pre_event': self.spin_pre_event.value(),
                'post_event': self.spin_post_event.value(),
            },
            'advanced': {
                'frame_skip': self.spin_frame_skip.value(),
                'max_cameras': self.spin_max_cameras.value(),
                'use_gpu': self.check_gpu.isChecked(),
                'log_level': self.combo_log_level.currentText(),
                'log_to_file': self.check_log_file.isChecked(),
            }
        }
