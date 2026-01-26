"""
Event Log Widget - Display detection events and system messages.
"""

from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QLabel, QFrame, QMenu
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QAction


class EventLogWidget(QWidget):
    """Widget for displaying detection events and system logs."""
    
    # Signals
    event_selected = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.events = []  # List of event dicts
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Filter bar
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "All Events",
            "Intrusions Only",
            "False Alarms",
            "System Messages"
        ])
        self.combo_filter.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.combo_filter)
        
        filter_layout.addWidget(QLabel("Camera:"))
        
        self.combo_camera = QComboBox()
        self.combo_camera.addItem("All Cameras")
        self.combo_camera.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.combo_camera)
        
        filter_layout.addStretch()
        
        self.btn_clear = QPushButton("Clear Log")
        self.btn_clear.clicked.connect(self._clear_log)
        filter_layout.addWidget(self.btn_clear)
        
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self._export_log)
        filter_layout.addWidget(self.btn_export)
        
        layout.addLayout(filter_layout)
        
        # Event table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Time", "Camera", "Type", "Objects", "Confidence", "Status"
        ])
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemDoubleClicked.connect(self._on_event_double_clicked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.table)
        
        # Statistics bar
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.StyledPanel)
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(5, 2, 5, 2)
        
        self.label_total = QLabel("Total: 0")
        stats_layout.addWidget(self.label_total)
        
        self.label_intrusions = QLabel("Intrusions: 0")
        self.label_intrusions.setStyleSheet("color: red;")
        stats_layout.addWidget(self.label_intrusions)
        
        self.label_filtered = QLabel("Filtered: 0")
        self.label_filtered.setStyleSheet("color: green;")
        stats_layout.addWidget(self.label_filtered)
        
        stats_layout.addStretch()
        
        layout.addWidget(stats_frame)
        
    def add_detection_event(self, camera_id: str, event_type: str, 
                           objects: list, confidence: float,
                           is_intrusion: bool, timestamp: datetime = None):
        """Add a detection event to the log."""
        if timestamp is None:
            timestamp = datetime.now()
            
        event = {
            'timestamp': timestamp,
            'camera_id': camera_id,
            'type': event_type,
            'objects': objects,
            'confidence': confidence,
            'is_intrusion': is_intrusion,
            'status': 'INTRUSION' if is_intrusion else 'Filtered'
        }
        
        self.events.insert(0, event)  # Add to beginning (newest first)
        self._add_event_to_table(event, 0)
        self._update_stats()
        
        # Update camera filter dropdown
        if camera_id not in [self.combo_camera.itemText(i) 
                             for i in range(self.combo_camera.count())]:
            self.combo_camera.addItem(camera_id)
            
    def add_system_event(self, message: str, timestamp: datetime = None):
        """Add a system event to the log."""
        if timestamp is None:
            timestamp = datetime.now()
            
        event = {
            'timestamp': timestamp,
            'camera_id': 'SYSTEM',
            'type': 'system',
            'objects': [],
            'confidence': 0,
            'is_intrusion': False,
            'status': message
        }
        
        self.events.insert(0, event)
        self._add_event_to_table(event, 0)
        
    def _add_event_to_table(self, event: dict, row: int = -1):
        """Add an event to the table."""
        if row < 0:
            row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Time
        time_str = event['timestamp'].strftime("%H:%M:%S")
        self.table.setItem(row, 0, QTableWidgetItem(time_str))
        
        # Camera
        self.table.setItem(row, 1, QTableWidgetItem(event['camera_id']))
        
        # Type
        self.table.setItem(row, 2, QTableWidgetItem(event['type']))
        
        # Objects
        objects_str = ", ".join(event['objects']) if event['objects'] else "-"
        self.table.setItem(row, 3, QTableWidgetItem(objects_str))
        
        # Confidence
        if event['confidence'] > 0:
            conf_str = f"{event['confidence']:.1%}"
        else:
            conf_str = "-"
        self.table.setItem(row, 4, QTableWidgetItem(conf_str))
        
        # Status
        status_item = QTableWidgetItem(event['status'])
        if event['is_intrusion']:
            status_item.setBackground(QColor(255, 200, 200))  # Light red
            status_item.setForeground(QColor(139, 0, 0))  # Dark red
        elif event['type'] == 'system':
            status_item.setForeground(QColor(0, 0, 139))  # Dark blue
        else:
            status_item.setForeground(QColor(0, 100, 0))  # Dark green
        self.table.setItem(row, 5, status_item)
        
    def _apply_filter(self):
        """Apply current filter settings."""
        filter_type = self.combo_filter.currentIndex()
        camera_filter = self.combo_camera.currentText()
        
        for row in range(self.table.rowCount()):
            show = True
            
            # Type filter
            if filter_type == 1:  # Intrusions only
                status = self.table.item(row, 5).text()
                show = status == 'INTRUSION'
            elif filter_type == 2:  # False alarms
                status = self.table.item(row, 5).text()
                show = status == 'Filtered'
            elif filter_type == 3:  # System messages
                event_type = self.table.item(row, 2).text()
                show = event_type == 'system'
                
            # Camera filter
            if show and camera_filter != "All Cameras":
                camera = self.table.item(row, 1).text()
                show = camera == camera_filter
                
            self.table.setRowHidden(row, not show)
            
    def _update_stats(self):
        """Update statistics labels."""
        total = len(self.events)
        intrusions = sum(1 for e in self.events if e['is_intrusion'])
        filtered = sum(1 for e in self.events 
                      if not e['is_intrusion'] and e['type'] != 'system')
        
        self.label_total.setText(f"Total: {total}")
        self.label_intrusions.setText(f"Intrusions: {intrusions}")
        self.label_filtered.setText(f"Filtered: {filtered}")
        
    def _clear_log(self):
        """Clear all events from the log."""
        self.events.clear()
        self.table.setRowCount(0)
        self._update_stats()
        
    def _export_log(self):
        """Export log to file."""
        # TODO: Implement export functionality
        pass
        
    def _on_event_double_clicked(self, item):
        """Handle double-click on event."""
        row = item.row()
        if 0 <= row < len(self.events):
            self.event_selected.emit(self.events[row])
            
    def _show_context_menu(self, pos):
        """Show context menu for event table."""
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
            
        menu = QMenu(self)
        
        action_view = QAction("View Details", self)
        action_view.triggered.connect(lambda: self._view_event_details(row))
        menu.addAction(action_view)
        
        action_export = QAction("Export Event", self)
        menu.addAction(action_export)
        
        menu.exec(self.table.mapToGlobal(pos))
        
    def _view_event_details(self, row):
        """View details for an event."""
        if 0 <= row < len(self.events):
            self.event_selected.emit(self.events[row])
            
    def get_event_count(self):
        """Return total number of events."""
        return len(self.events)
        
    def get_intrusion_count(self):
        """Return number of detected intrusions."""
        return sum(1 for e in self.events if e['is_intrusion'])
