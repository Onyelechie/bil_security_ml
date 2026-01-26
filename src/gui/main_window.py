"""
Main Window for BIL Security ML Application.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QStatusBar, QToolBar, QMessageBox,
    QLabel, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QIcon

from .camera_panel import CameraPanel
from .event_log import EventLogWidget
from .settings_dialog import SettingsDialog
from .test_panel import TestPanel


class MainWindow(QMainWindow):
    """Main application window for intrusion detection system."""
    
    # Signals
    detection_started = Signal()
    detection_stopped = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIL Security ML - Intrusion Detection System")
        self.setMinimumSize(1200, 800)
        
        # State
        self.is_running = False
        self.cameras = {}  # camera_id -> camera config
        
        # Setup UI
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_statusbar()
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)  # Update every second
        
    def _setup_menubar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        self.action_start = QAction("&Start Detection", self)
        self.action_start.setShortcut("Ctrl+S")
        self.action_start.triggered.connect(self.start_detection)
        file_menu.addAction(self.action_start)
        
        self.action_stop = QAction("S&top Detection", self)
        self.action_stop.setShortcut("Ctrl+T")
        self.action_stop.setEnabled(False)
        self.action_stop.triggered.connect(self.stop_detection)
        file_menu.addAction(self.action_stop)
        
        file_menu.addSeparator()
        
        action_exit = QAction("E&xit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)
        
        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        
        action_settings = QAction("&Preferences...", self)
        action_settings.setShortcut("Ctrl+,")
        action_settings.triggered.connect(self._show_settings)
        settings_menu.addAction(action_settings)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        action_about = QAction("&About", self)
        action_about.triggered.connect(self._show_about)
        help_menu.addAction(action_about)
        
    def _setup_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Start/Stop buttons
        self.btn_start = QAction("▶ Start", self)
        self.btn_start.triggered.connect(self.start_detection)
        toolbar.addAction(self.btn_start)
        
        self.btn_stop = QAction("⬛ Stop", self)
        self.btn_stop.setEnabled(False)
        self.btn_stop.triggered.connect(self.stop_detection)
        toolbar.addAction(self.btn_stop)
        
        toolbar.addSeparator()
        
        # Settings button
        btn_settings = QAction("⚙ Settings", self)
        btn_settings.triggered.connect(self._show_settings)
        toolbar.addAction(btn_settings)
        
    def _setup_central_widget(self):
        """Create the central widget with main layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget for different sections
        self.tabs = QTabWidget()
        
        # === Tab 1: Cameras & Events ===
        cameras_tab = QWidget()
        cameras_layout = QHBoxLayout(cameras_tab)
        cameras_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Camera management
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        
        left_label = QLabel("<b>Cameras</b>")
        left_layout.addWidget(left_label)
        
        self.camera_panel = CameraPanel()
        left_layout.addWidget(self.camera_panel)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Event log and status
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        
        right_label = QLabel("<b>Detection Events</b>")
        right_layout.addWidget(right_label)
        
        self.event_log = EventLogWidget()
        right_layout.addWidget(self.event_log)
        
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (40% cameras, 60% events)
        splitter.setSizes([400, 600])
        
        cameras_layout.addWidget(splitter)
        self.tabs.addTab(cameras_tab, "📹 Cameras & Events")
        
        # === Tab 2: Test ===
        self.test_panel = TestPanel()
        self.tabs.addTab(self.test_panel, "🧪 Test")
        
        layout.addWidget(self.tabs)
        
    def _setup_statusbar(self):
        """Create the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label)
        
        # Spacer
        self.statusbar.addPermanentWidget(QWidget(), 1)
        
        # Camera count
        self.camera_count_label = QLabel("Cameras: 0")
        self.statusbar.addPermanentWidget(self.camera_count_label)
        
        # Event count
        self.event_count_label = QLabel("Events: 0")
        self.statusbar.addPermanentWidget(self.event_count_label)
        
    @Slot()
    def start_detection(self):
        """Start the detection pipeline."""
        if self.is_running:
            return
            
        # Check if any cameras are configured
        if self.camera_panel.get_camera_count() == 0:
            QMessageBox.warning(
                self, 
                "No Cameras",
                "Please add at least one camera before starting detection."
            )
            return
            
        self.is_running = True
        self.action_start.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Detection Running...")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.detection_started.emit()
        self.event_log.add_system_event("Detection started")
        
    @Slot()
    def stop_detection(self):
        """Stop the detection pipeline."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.action_start.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: red;")
        
        self.detection_stopped.emit()
        self.event_log.add_system_event("Detection stopped")
        
    @Slot()
    def _show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Settings were saved
            self.event_log.add_system_event("Settings updated")
            
    @Slot()
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About BIL Security ML",
            "<h3>BIL Security ML</h3>"
            "<p>Intrusion Detection System v1.0</p>"
            "<p>Event-driven intelligent video analysis for security cameras.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>YOLOv8 object detection</li>"
            "<li>Motion-based false alarm filtering</li>"
            "<li>Multi-camera support (up to 10)</li>"
            "<li>ONVIF/RTSP compatible</li>"
            "</ul>"
            "<p>© 2026 COMP 4560 Industrial Project</p>"
        )
        
    @Slot()
    def _update_status(self):
        """Update status bar information."""
        cam_count = self.camera_panel.get_camera_count()
        event_count = self.event_log.get_event_count()
        
        self.camera_count_label.setText(f"Cameras: {cam_count}")
        self.event_count_label.setText(f"Events: {event_count}")
        
    def closeEvent(self, event):
        """Handle window close event."""
        if self.is_running:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Detection is currently running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
                
        # Stop detection before closing
        self.stop_detection()
        event.accept()
