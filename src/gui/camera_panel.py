"""
Camera Panel - Manage camera configurations.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QFormLayout, QLineEdit, QComboBox,
    QSpinBox, QDialogButtonBox, QMessageBox, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction

from ..detect.config import AVAILABLE_MODELS, DEFAULT_MODEL


class CameraConfigDialog(QDialog):
    """Dialog for adding/editing camera configuration."""
    
    def __init__(self, parent=None, camera_data=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Configuration")
        self.setMinimumWidth(400)
        
        self.camera_data = camera_data or {}
        self._setup_ui()
        
        if camera_data:
            self._load_data(camera_data)
            
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        # Camera ID
        self.edit_id = QLineEdit()
        self.edit_id.setPlaceholderText("e.g., cam_01")
        form.addRow("Camera ID:", self.edit_id)
        
        # Camera Name
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g., Front Entrance")
        form.addRow("Name:", self.edit_name)
        
        # Source (RTSP URL or video file)
        self.edit_source = QLineEdit()
        self.edit_source.setPlaceholderText("rtsp://... or path/to/video.mp4")
        form.addRow("Source:", self.edit_source)
        
        # Model selection
        self.combo_model = QComboBox()
        for model_key in AVAILABLE_MODELS:
            self.combo_model.addItem(model_key)
        # Set default
        default_idx = self.combo_model.findText(DEFAULT_MODEL)
        if default_idx >= 0:
            self.combo_model.setCurrentIndex(default_idx)
        form.addRow("Detection Model:", self.combo_model)
        
        # Camera type hint (affects model recommendation)
        self.combo_type = QComboBox()
        self.combo_type.addItems([
            "Wide Angle (Parking Lot, Building Exterior)",
            "Close Range (Entry Point, Doorway)",
            "Indoor (Hallway, Room)"
        ])
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("Camera Type:", self.combo_type)
        
        # Confidence threshold
        self.spin_confidence = QSpinBox()
        self.spin_confidence.setRange(10, 100)
        self.spin_confidence.setValue(50)
        self.spin_confidence.setSuffix("%")
        form.addRow("Confidence Threshold:", self.spin_confidence)
        
        layout.addLayout(form)
        
        # Model recommendation label
        self.label_recommendation = QLineEdit()
        self.label_recommendation.setReadOnly(True)
        self.label_recommendation.setStyleSheet("color: gray; font-style: italic;")
        self._update_recommendation()
        layout.addWidget(self.label_recommendation)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def _on_type_changed(self, index):
        """Handle camera type change - suggest appropriate model."""
        self._update_recommendation()
        
    def _update_recommendation(self):
        """Update model recommendation based on camera type."""
        type_idx = self.combo_type.currentIndex()
        if type_idx == 0:  # Wide angle
            rec = "Recommended: yolov8s (better accuracy for distant objects)"
        elif type_idx == 1:  # Close range
            rec = "Recommended: yolov8n (faster, good for close-up detection)"
        else:  # Indoor
            rec = "Recommended: yolov8n or mobilenet (balance of speed/accuracy)"
        self.label_recommendation.setText(rec)
        
    def _load_data(self, data):
        """Load existing camera data into form."""
        self.edit_id.setText(data.get('id', ''))
        self.edit_id.setEnabled(False)  # Can't change ID when editing
        self.edit_name.setText(data.get('name', ''))
        self.edit_source.setText(data.get('source', ''))
        
        model_idx = self.combo_model.findText(data.get('model', DEFAULT_MODEL))
        if model_idx >= 0:
            self.combo_model.setCurrentIndex(model_idx)
            
        self.spin_confidence.setValue(int(data.get('confidence', 50) * 100))
        
    def _validate_and_accept(self):
        """Validate input before accepting."""
        if not self.edit_id.text().strip():
            QMessageBox.warning(self, "Validation Error", "Camera ID is required.")
            return
        if not self.edit_source.text().strip():
            QMessageBox.warning(self, "Validation Error", "Source URL/path is required.")
            return
        self.accept()
        
    def get_camera_data(self):
        """Return the configured camera data."""
        return {
            'id': self.edit_id.text().strip(),
            'name': self.edit_name.text().strip() or self.edit_id.text().strip(),
            'source': self.edit_source.text().strip(),
            'model': self.combo_model.currentText(),
            'confidence': self.spin_confidence.value() / 100.0,
            'enabled': True
        }


class CameraPanel(QWidget):
    """Panel for managing cameras."""
    
    # Signals
    camera_added = Signal(dict)
    camera_removed = Signal(str)
    camera_updated = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.cameras = {}  # id -> camera_data
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("+ Add Camera")
        self.btn_add.clicked.connect(self._add_camera)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_edit = QPushButton("Edit")
        self.btn_edit.setEnabled(False)
        self.btn_edit.clicked.connect(self._edit_camera)
        btn_layout.addWidget(self.btn_edit)
        
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._remove_camera)
        btn_layout.addWidget(self.btn_remove)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Camera table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "ID", "Name", "Model", "Status", "Events"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.table)
        
    def _add_camera(self):
        """Show dialog to add a new camera."""
        # Check camera limit
        if len(self.cameras) >= 10:
            QMessageBox.warning(
                self,
                "Camera Limit",
                "Maximum of 10 cameras supported per device."
            )
            return
            
        dialog = CameraConfigDialog(self)
        if dialog.exec():
            data = dialog.get_camera_data()
            
            # Check for duplicate ID
            if data['id'] in self.cameras:
                QMessageBox.warning(
                    self,
                    "Duplicate ID",
                    f"Camera with ID '{data['id']}' already exists."
                )
                return
                
            self.cameras[data['id']] = data
            self._refresh_table()
            self.camera_added.emit(data)
            
    def _edit_camera(self):
        """Edit selected camera."""
        row = self.table.currentRow()
        if row < 0:
            return
            
        cam_id = self.table.item(row, 0).text()
        data = self.cameras.get(cam_id)
        if not data:
            return
            
        dialog = CameraConfigDialog(self, data)
        if dialog.exec():
            new_data = dialog.get_camera_data()
            self.cameras[cam_id] = new_data
            self._refresh_table()
            self.camera_updated.emit(new_data)
            
    def _remove_camera(self):
        """Remove selected camera."""
        row = self.table.currentRow()
        if row < 0:
            return
            
        cam_id = self.table.item(row, 0).text()
        
        reply = QMessageBox.question(
            self,
            "Confirm Remove",
            f"Remove camera '{cam_id}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.cameras[cam_id]
            self._refresh_table()
            self.camera_removed.emit(cam_id)
            
    def _refresh_table(self):
        """Refresh the camera table."""
        self.table.setRowCount(len(self.cameras))
        
        for row, (cam_id, data) in enumerate(self.cameras.items()):
            self.table.setItem(row, 0, QTableWidgetItem(cam_id))
            self.table.setItem(row, 1, QTableWidgetItem(data.get('name', '')))
            self.table.setItem(row, 2, QTableWidgetItem(data.get('model', '')))
            
            status = "Ready" if data.get('enabled', True) else "Disabled"
            status_item = QTableWidgetItem(status)
            status_item.setForeground(
                Qt.green if status == "Ready" else Qt.gray
            )
            self.table.setItem(row, 3, status_item)
            
            self.table.setItem(row, 4, QTableWidgetItem("0"))
            
    def _on_selection_changed(self):
        """Handle table selection change."""
        has_selection = self.table.currentRow() >= 0
        self.btn_edit.setEnabled(has_selection)
        self.btn_remove.setEnabled(has_selection)
        
    def _show_context_menu(self, pos):
        """Show context menu for camera table."""
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
            
        menu = QMenu(self)
        
        action_edit = QAction("Edit", self)
        action_edit.triggered.connect(self._edit_camera)
        menu.addAction(action_edit)
        
        action_remove = QAction("Remove", self)
        action_remove.triggered.connect(self._remove_camera)
        menu.addAction(action_remove)
        
        menu.exec(self.table.mapToGlobal(pos))
        
    def get_camera_count(self):
        """Return number of configured cameras."""
        return len(self.cameras)
        
    def get_cameras(self):
        """Return all camera configurations."""
        return dict(self.cameras)
        
    def update_camera_status(self, cam_id, status, event_count=None):
        """Update camera status in table."""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == cam_id:
                status_item = QTableWidgetItem(status)
                if status == "Running":
                    status_item.setForeground(Qt.green)
                elif status == "Error":
                    status_item.setForeground(Qt.red)
                else:
                    status_item.setForeground(Qt.gray)
                self.table.setItem(row, 3, status_item)
                
                if event_count is not None:
                    self.table.setItem(row, 4, QTableWidgetItem(str(event_count)))
                break
