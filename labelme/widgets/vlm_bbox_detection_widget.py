"""
VLM Bounding Box Detection Widget.

This widget provides a UI for AI-powered object detection using VLM.
Users can enter object names and get automatic bounding box detection.
"""

from loguru import logger
from PyQt5 import QtWidgets
from typing import Callable


class VlmBboxDetectionWidget(QtWidgets.QWidget):
    """
    Widget for VLM-based object detection and bounding box generation.
    
    This widget replaces the old AiLabelWidget with a clearer name and purpose.
    It provides a text input for object names and a submit button for detection.
    """
    
    def __init__(self, on_detect_callback: Callable[[], None], parent=None):
        """
        Initialize the VLM bbox detection widget.
        
        Args:
            on_detect_callback: Function to call when detection is requested
            parent: Parent Qt widget
        """
        super().__init__(parent=parent)
        self._setup_ui(on_detect_callback)

    def _setup_ui(self, on_detect_callback: Callable[[], None]) -> None:
        """Set up the user interface components."""
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(0)  # type: ignore[union-attr]

        # Create the main input widget
        input_widget = _ObjectDetectionInputWidget(
            on_detect_callback=on_detect_callback, 
            parent=self
        )
        input_widget.setMaximumWidth(400)
        self.layout().addWidget(input_widget)  # type: ignore[union-attr]

    def get_object_names(self) -> str:
        """
        Get the current object names from the input field.
        
        Returns:
            Comma-separated object names entered by the user
        """
        layout = self.layout()
        if not layout:
            logger.warning("Widget layout not found")
            return ""
            
        item = layout.itemAt(0)
        if not item or not item.widget():
            logger.warning("Input widget not found")
            return ""
            
        return item.widget().get_object_names()
    
    def set_object_names(self, objects: str) -> None:
        """
        Set the object names in the input field.
        
        Args:
            objects: Comma-separated object names to populate
        """
        layout = self.layout()
        if not layout:
            return
            
        item = layout.itemAt(0)
        if not item or not item.widget():
            return
            
        item.widget().set_object_names(objects)
        
    def clear_input(self) -> None:
        """Clear the object names input field."""
        self.set_object_names("")


class _ObjectDetectionInputWidget(QtWidgets.QWidget):
    """
    Internal widget for object detection input controls.
    
    Contains label, text input, and submit button in horizontal layout.
    """
    
    def __init__(self, on_detect_callback: Callable[[], None], parent=None):
        """
        Initialize the input widget.
        
        Args:
            on_detect_callback: Function to call when detection is requested
            parent: Parent Qt widget
        """
        super().__init__(parent=parent)
        self._setup_ui(on_detect_callback)

    def _setup_ui(self, on_detect_callback: Callable[[], None]) -> None:
        """Set up the horizontal layout with controls."""
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)  # type: ignore[union-attr]

        # Label
        label = QtWidgets.QLabel(self.tr("Objects to Detect"))
        label.setToolTip(self.tr("Enter object names separated by commas"))
        self.layout().addWidget(label)  # type: ignore[union-attr]

        # Text input field
        self._text_input = QtWidgets.QLineEdit()
        self._text_input.setPlaceholderText(self.tr("e.g., dog, cat, bird, person"))
        self._text_input.setToolTip(
            self.tr("Enter the names of objects you want to detect, separated by commas")
        )
        
        # Connect Enter key to detection
        self._text_input.returnPressed.connect(on_detect_callback)
        self.layout().addWidget(self._text_input)  # type: ignore[union-attr]

        # Submit button
        detect_button = QtWidgets.QPushButton(text=self.tr("Detect Objects"), parent=self)
        detect_button.setToolTip(self.tr("Run AI object detection on the current image"))
        detect_button.clicked.connect(on_detect_callback)
        self.layout().addWidget(detect_button)  # type: ignore[union-attr]

    def get_object_names(self) -> str:
        """Get the current text from the input field."""
        return self._text_input.text().strip()

    def set_object_names(self, objects: str) -> None:
        """Set the text in the input field."""
        self._text_input.setText(objects)
        
    def focus_input(self) -> None:
        """Set focus to the text input field."""
        self._text_input.setFocus()
        
    def select_all_text(self) -> None:
        """Select all text in the input field."""
        self._text_input.selectAll() 