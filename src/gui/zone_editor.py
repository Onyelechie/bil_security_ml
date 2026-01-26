"""
Zone Editor - Define custom monitoring regions for cameras.

Allows users to draw polygon zones by clicking points.
Only detections within these zones will trigger alerts.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QDialog,
    QDialogButtonBox, QLineEdit, QColorDialog, QMessageBox,
    QFrame, QSplitter, QMenu, QInputDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QPoint, QRect
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor,
    QPolygon, QMouseEvent, QPaintEvent
)


@dataclass
class Zone:
    """Represents a monitoring zone (polygon region)."""
    name: str
    points: List[Tuple[int, int]]  # List of (x, y) points
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR color
    enabled: bool = True
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside the zone polygon."""
        if len(self.points) < 3:
            return False
        # Use cv2.pointPolygonTest
        contour = np.array(self.points, dtype=np.int32)
        result = cv2.pointPolygonTest(contour, (x, y), False)
        return result >= 0
    
    def contains_box(self, bbox: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
        """
        Check if a bounding box overlaps with the zone.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            threshold: Minimum overlap ratio to consider "inside"
        
        Returns:
            True if the box center or sufficient area is inside the zone
        """
        if len(self.points) < 3:
            return False
            
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Check if center is inside
        if self.contains_point(center_x, center_y):
            return True
            
        # Check corners
        corners_inside = 0
        for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            if self.contains_point(px, py):
                corners_inside += 1
                
        # If any corner is inside, consider it overlapping
        return corners_inside > 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'points': self.points,
            'color': self.color,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Zone':
        """Create Zone from dictionary."""
        return cls(
            name=data['name'],
            points=[tuple(p) for p in data['points']],
            color=tuple(data.get('color', (0, 255, 0))),
            enabled=data.get('enabled', True)
        )


class ZoneManager:
    """Manages zones for multiple cameras."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.zones: Dict[str, List[Zone]] = {}  # camera_id -> list of zones
        self.config_path = config_path
        
        if config_path and config_path.exists():
            self.load()
    
    def add_zone(self, camera_id: str, zone: Zone):
        """Add a zone for a camera."""
        if camera_id not in self.zones:
            self.zones[camera_id] = []
        self.zones[camera_id].append(zone)
        
    def remove_zone(self, camera_id: str, zone_name: str):
        """Remove a zone by name."""
        if camera_id in self.zones:
            self.zones[camera_id] = [z for z in self.zones[camera_id] 
                                     if z.name != zone_name]
    
    def get_zones(self, camera_id: str) -> List[Zone]:
        """Get all zones for a camera."""
        return self.zones.get(camera_id, [])
    
    def get_enabled_zones(self, camera_id: str) -> List[Zone]:
        """Get only enabled zones for a camera."""
        return [z for z in self.get_zones(camera_id) if z.enabled]
    
    def is_in_any_zone(self, camera_id: str, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if a detection box is in any enabled zone."""
        zones = self.get_enabled_zones(camera_id)
        if not zones:
            return True  # No zones defined = entire frame monitored
        return any(z.contains_box(bbox) for z in zones)
    
    def filter_detections(self, camera_id: str, detections: List[dict]) -> List[dict]:
        """Filter detections to only those within enabled zones."""
        zones = self.get_enabled_zones(camera_id)
        if not zones:
            return detections  # No zones = all detections pass
            
        filtered = []
        for det in detections:
            bbox = det.get('box') or det.get('bbox')
            if bbox and any(z.contains_box(bbox) for z in zones):
                det['in_zone'] = True
                filtered.append(det)
            else:
                det['in_zone'] = False
        return filtered
    
    def save(self, path: Optional[Path] = None):
        """Save zones to JSON file."""
        path = path or self.config_path
        if not path:
            return
            
        data = {}
        for camera_id, zones in self.zones.items():
            data[camera_id] = [z.to_dict() for z in zones]
            
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Optional[Path] = None):
        """Load zones from JSON file."""
        path = path or self.config_path
        if not path or not path.exists():
            return
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.zones = {}
        for camera_id, zone_list in data.items():
            self.zones[camera_id] = [Zone.from_dict(z) for z in zone_list]


class ZoneCanvas(QLabel):
    """Canvas widget for drawing zones on a video frame."""
    
    zone_completed = Signal(list)  # Emits list of points when zone is done
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMouseTracking(True)
        
        # State
        self.background_image: Optional[QPixmap] = None
        self.current_points: List[QPoint] = []
        self.existing_zones: List[Zone] = []
        self.is_drawing = False
        self.hover_point: Optional[QPoint] = None
        
        # Scale factor for coordinate conversion
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
    def set_background(self, frame: np.ndarray):
        """Set the background image from a video frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.background_image = QPixmap.fromImage(q_img)
        
        # Calculate scale factors
        self._update_scale()
        self.update()
        
    def set_background_pixmap(self, pixmap: QPixmap):
        """Set background from QPixmap."""
        self.background_image = pixmap
        self._update_scale()
        self.update()
        
    def _update_scale(self):
        """Update scale factors for coordinate conversion."""
        if not self.background_image:
            return
            
        img_w = self.background_image.width()
        img_h = self.background_image.height()
        widget_w = self.width()
        widget_h = self.height()
        
        # Calculate scaled size maintaining aspect ratio
        scale = min(widget_w / img_w, widget_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        
        self.scale_x = img_w / scaled_w
        self.scale_y = img_h / scaled_h
        self.offset_x = (widget_w - scaled_w) // 2
        self.offset_y = (widget_h - scaled_h) // 2
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        
    def set_zones(self, zones: List[Zone]):
        """Set existing zones to display."""
        self.existing_zones = zones
        self.update()
        
    def start_drawing(self):
        """Start drawing a new zone."""
        self.is_drawing = True
        self.current_points = []
        self.setCursor(Qt.CrossCursor)
        self.update()
        
    def finish_drawing(self):
        """Finish drawing and emit the zone."""
        if len(self.current_points) >= 3:
            # Convert to image coordinates
            points = []
            for p in self.current_points:
                img_x = int((p.x() - self.offset_x) * self.scale_x)
                img_y = int((p.y() - self.offset_y) * self.scale_y)
                points.append((img_x, img_y))
            self.zone_completed.emit(points)
            
        self.is_drawing = False
        self.current_points = []
        self.setCursor(Qt.ArrowCursor)
        self.update()
        
    def cancel_drawing(self):
        """Cancel current drawing."""
        self.is_drawing = False
        self.current_points = []
        self.setCursor(Qt.ArrowCursor)
        self.update()
        
    def undo_last_point(self):
        """Remove the last point."""
        if self.current_points:
            self.current_points.pop()
            self.update()
            
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse clicks for adding points."""
        if not self.is_drawing:
            return
            
        if event.button() == Qt.LeftButton:
            # Add point
            self.current_points.append(event.position().toPoint())
            self.update()
        elif event.button() == Qt.RightButton:
            # Finish drawing
            self.finish_drawing()
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Track mouse for preview."""
        if self.is_drawing:
            self.hover_point = event.position().toPoint()
            self.update()
            
    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        self._update_scale()
        
    def paintEvent(self, event: QPaintEvent):
        """Draw the canvas with zones."""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background image
        if self.background_image:
            scaled = self.background_image.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            
        # Draw existing zones
        for zone in self.existing_zones:
            if len(zone.points) >= 3:
                # Convert image coords to widget coords
                widget_points = []
                for px, py in zone.points:
                    wx = int(px / self.scale_x) + self.offset_x
                    wy = int(py / self.scale_y) + self.offset_y
                    widget_points.append(QPoint(wx, wy))
                
                polygon = QPolygon(widget_points)
                
                # Fill with semi-transparent color
                color = QColor(zone.color[2], zone.color[1], zone.color[0], 50)
                painter.setBrush(QBrush(color))
                
                # Border
                border_color = QColor(zone.color[2], zone.color[1], zone.color[0])
                if not zone.enabled:
                    border_color = QColor(128, 128, 128)
                pen = QPen(border_color, 2)
                painter.setPen(pen)
                
                painter.drawPolygon(polygon)
                
                # Draw zone name
                if widget_points:
                    center_x = sum(p.x() for p in widget_points) // len(widget_points)
                    center_y = sum(p.y() for p in widget_points) // len(widget_points)
                    painter.setPen(QPen(Qt.white))
                    painter.drawText(center_x - 30, center_y, zone.name)
                    
        # Draw current drawing
        if self.is_drawing and self.current_points:
            # Draw points
            painter.setPen(QPen(Qt.red, 2))
            painter.setBrush(QBrush(Qt.red))
            
            for i, point in enumerate(self.current_points):
                painter.drawEllipse(point, 5, 5)
                
                # Draw lines between points
                if i > 0:
                    painter.drawLine(self.current_points[i-1], point)
                    
            # Draw preview line to cursor
            if self.hover_point and self.current_points:
                painter.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                painter.drawLine(self.current_points[-1], self.hover_point)
                
                # Draw closing line preview
                if len(self.current_points) >= 2:
                    painter.drawLine(self.hover_point, self.current_points[0])
                    
        # Draw instructions
        if self.is_drawing:
            painter.setPen(QPen(Qt.white))
            painter.drawText(10, 20, f"Points: {len(self.current_points)} | Left-click: Add point | Right-click: Finish | Esc: Cancel")
        
        painter.end()


class ZoneEditorDialog(QDialog):
    """Dialog for editing zones for a camera or test."""
    
    def __init__(self, parent=None, camera_id: str = "test", 
                 initial_frame: Optional[np.ndarray] = None,
                 existing_zones: Optional[List[Zone]] = None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.zones = list(existing_zones) if existing_zones else []
        
        self.setWindowTitle(f"Zone Editor - {camera_id}")
        self.setMinimumSize(900, 700)
        
        self._setup_ui()
        
        if initial_frame is not None:
            self.canvas.set_background(initial_frame)
        self.canvas.set_zones(self.zones)
        
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QHBoxLayout(self)
        
        # Left: Canvas
        self.canvas = ZoneCanvas()
        self.canvas.zone_completed.connect(self._on_zone_completed)
        layout.addWidget(self.canvas, stretch=3)
        
        # Right: Controls
        controls = QFrame()
        controls.setFrameStyle(QFrame.StyledPanel)
        controls.setMaximumWidth(250)
        controls_layout = QVBoxLayout(controls)
        
        # Zone list
        controls_layout.addWidget(QLabel("<b>Zones</b>"))
        
        self.zone_list = QListWidget()
        self.zone_list.itemSelectionChanged.connect(self._on_zone_selected)
        self.zone_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.zone_list.customContextMenuRequested.connect(self._show_zone_menu)
        controls_layout.addWidget(self.zone_list)
        
        self._refresh_zone_list()
        
        # Zone buttons
        btn_layout = QHBoxLayout()
        
        self.btn_new = QPushButton("+ New Zone")
        self.btn_new.clicked.connect(self._start_new_zone)
        btn_layout.addWidget(self.btn_new)
        
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._delete_selected_zone)
        btn_layout.addWidget(self.btn_delete)
        
        controls_layout.addLayout(btn_layout)
        
        # Drawing controls (shown when drawing)
        self.drawing_frame = QFrame()
        drawing_layout = QVBoxLayout(self.drawing_frame)
        drawing_layout.setContentsMargins(0, 10, 0, 0)
        
        drawing_layout.addWidget(QLabel("<b>Drawing Mode</b>"))
        drawing_layout.addWidget(QLabel("• Left-click to add points"))
        drawing_layout.addWidget(QLabel("• Right-click to finish"))
        drawing_layout.addWidget(QLabel("• Need at least 3 points"))
        
        btn_drawing_layout = QHBoxLayout()
        
        self.btn_undo = QPushButton("Undo Point")
        self.btn_undo.clicked.connect(self.canvas.undo_last_point)
        btn_drawing_layout.addWidget(self.btn_undo)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_drawing)
        btn_drawing_layout.addWidget(self.btn_cancel)
        
        drawing_layout.addLayout(btn_drawing_layout)
        
        self.btn_done_drawing = QPushButton("✓ Done (3+ points)")
        self.btn_done_drawing.clicked.connect(self.canvas.finish_drawing)
        drawing_layout.addWidget(self.btn_done_drawing)
        
        self.drawing_frame.hide()
        controls_layout.addWidget(self.drawing_frame)
        
        controls_layout.addStretch()
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        controls_layout.addWidget(buttons)
        
        layout.addWidget(controls)
        
    def _refresh_zone_list(self):
        """Refresh the zone list widget."""
        self.zone_list.clear()
        for zone in self.zones:
            item = QListWidgetItem(f"{'✓' if zone.enabled else '○'} {zone.name}")
            item.setData(Qt.UserRole, zone.name)
            self.zone_list.addItem(item)
            
    def _start_new_zone(self):
        """Start drawing a new zone."""
        self.btn_new.setEnabled(False)
        self.drawing_frame.show()
        self.canvas.start_drawing()
        
    def _cancel_drawing(self):
        """Cancel current drawing."""
        self.canvas.cancel_drawing()
        self.btn_new.setEnabled(True)
        self.drawing_frame.hide()
        
    @Slot(list)
    def _on_zone_completed(self, points: List[Tuple[int, int]]):
        """Handle completed zone drawing."""
        if len(points) < 3:
            QMessageBox.warning(self, "Invalid Zone", "Zone needs at least 3 points.")
            return
            
        # Get zone name
        name, ok = QInputDialog.getText(
            self, "Zone Name", "Enter a name for this zone:",
            text=f"Zone {len(self.zones) + 1}"
        )
        
        if ok and name:
            # Pick a color
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 165, 255),  # Orange
                (255, 255, 0),  # Cyan
            ]
            color = colors[len(self.zones) % len(colors)]
            
            zone = Zone(name=name, points=points, color=color)
            self.zones.append(zone)
            self._refresh_zone_list()
            self.canvas.set_zones(self.zones)
            
        self.btn_new.setEnabled(True)
        self.drawing_frame.hide()
        
    def _on_zone_selected(self):
        """Handle zone selection."""
        has_selection = self.zone_list.currentRow() >= 0
        self.btn_delete.setEnabled(has_selection)
        
    def _delete_selected_zone(self):
        """Delete the selected zone."""
        row = self.zone_list.currentRow()
        if row >= 0 and row < len(self.zones):
            zone = self.zones[row]
            reply = QMessageBox.question(
                self, "Delete Zone",
                f"Delete zone '{zone.name}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                del self.zones[row]
                self._refresh_zone_list()
                self.canvas.set_zones(self.zones)
                
    def _show_zone_menu(self, pos):
        """Show context menu for zones."""
        item = self.zone_list.itemAt(pos)
        if not item:
            return
            
        row = self.zone_list.row(item)
        if row < 0 or row >= len(self.zones):
            return
            
        zone = self.zones[row]
        
        menu = QMenu(self)
        
        # Toggle enabled
        action_toggle = menu.addAction(
            "Disable" if zone.enabled else "Enable"
        )
        action_toggle.triggered.connect(lambda: self._toggle_zone(row))
        
        # Rename
        action_rename = menu.addAction("Rename")
        action_rename.triggered.connect(lambda: self._rename_zone(row))
        
        # Delete
        action_delete = menu.addAction("Delete")
        action_delete.triggered.connect(self._delete_selected_zone)
        
        menu.exec(self.zone_list.mapToGlobal(pos))
        
    def _toggle_zone(self, row: int):
        """Toggle zone enabled state."""
        if 0 <= row < len(self.zones):
            self.zones[row].enabled = not self.zones[row].enabled
            self._refresh_zone_list()
            self.canvas.set_zones(self.zones)
            
    def _rename_zone(self, row: int):
        """Rename a zone."""
        if 0 <= row < len(self.zones):
            zone = self.zones[row]
            name, ok = QInputDialog.getText(
                self, "Rename Zone", "Enter new name:",
                text=zone.name
            )
            if ok and name:
                zone.name = name
                self._refresh_zone_list()
                self.canvas.set_zones(self.zones)
                
    def get_zones(self) -> List[Zone]:
        """Return the configured zones."""
        return self.zones
    
    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key_Escape and self.canvas.is_drawing:
            self._cancel_drawing()
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.canvas.undo_last_point()
        else:
            super().keyPressEvent(event)
