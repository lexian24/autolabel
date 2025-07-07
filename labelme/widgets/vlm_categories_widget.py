"""
VLM Categories Widget.

This widget provides an organized dropdown interface for VLM functionalities:
- Image (Grounding, Caption)
- Video (placeholder)
- Audio (placeholder)
"""

from loguru import logger
from PyQt5 import QtWidgets, QtCore
from typing import Callable, List, Dict, Optional


class VlmCategoriesWidget(QtWidgets.QWidget):
    """
    Widget for organizing VLM functions into categories with dropdowns.
    
    Provides a hierarchical interface:
    - Image -> Grounding (Object Detection, AI Labeling)
    - Image -> Caption (Image Captioning, Region Captioning)
    - Video (placeholder)
    - Audio (placeholder)
    """
    
    # Signals for different operations
    caption_requested = QtCore.pyqtSignal(str)   # prompt text for caption
    subcategory_changed = QtCore.pyqtSignal(str)  # signal when image subcategory changes
    auto_label_requested = QtCore.pyqtSignal(str, str)  # prompt text, task type (Detection/OCR)
    
    def __init__(self, parent=None):
        """Initialize the VLM categories widget."""
        super().__init__(parent=parent)
        self._setup_ui()
        self._prompt_history: List[Dict[str, str]] = []
        
    def _setup_ui(self) -> None:
        """Set up the user interface components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Main category dropdown
        self._category_combo = QtWidgets.QComboBox()
        self._category_combo.addItems(["Image", "Video", "Audio"])
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        layout.addWidget(self._category_combo)
        
        # Stacked widget for different category interfaces
        self._stacked_widget = QtWidgets.QStackedWidget()
        layout.addWidget(self._stacked_widget)
        
        # Setup category-specific interfaces
        self._setup_image_interface()
        self._setup_video_interface()
        self._setup_audio_interface()
        
        # Set default to Image
        self._category_combo.setCurrentText("Image")
        
    def _setup_image_interface(self) -> None:
        """Setup the Image category interface with Grounding and Caption."""
        image_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(image_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Subcategory dropdown
        subcategory_combo = QtWidgets.QComboBox()
        subcategory_combo.addItems(["Detection", "Caption", "OCR"])
        subcategory_combo.currentTextChanged.connect(self._on_image_subcategory_changed)
        layout.addWidget(subcategory_combo)
        
        # Stacked widget for subcategories
        subcategory_stack = QtWidgets.QStackedWidget()
        layout.addWidget(subcategory_stack)
        
        # Detection interface (standard labelme annotation)
        detection_widget = self._create_detection_interface()
        subcategory_stack.addWidget(detection_widget)
        
        # Caption interface (with prompt-output pairs)
        caption_widget = self._create_caption_interface()
        subcategory_stack.addWidget(caption_widget)
        
        # OCR interface (standard labelme annotation)
        ocr_widget = self._create_ocr_interface()
        subcategory_stack.addWidget(ocr_widget)
        
        # Store references
        image_widget.subcategory_combo = subcategory_combo
        image_widget.subcategory_stack = subcategory_stack
        
        self._stacked_widget.addWidget(image_widget)
        
    def _create_detection_interface(self) -> QtWidgets.QWidget:
        """Create interface for detection tasks using standard labelme annotation."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QtWidgets.QLabel("Object Detection")
        title_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # VLM Auto Labeling Section
        vlm_label = QtWidgets.QLabel("VLM Auto Labeling:")
        vlm_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(vlm_label)
        
        # Prompt input
        self._detection_prompt = QtWidgets.QLineEdit()
        self._detection_prompt.setPlaceholderText("Enter objects to detect (e.g., person, car, dog)")
        layout.addWidget(self._detection_prompt)
        
        # Auto label button
        detection_btn = QtWidgets.QPushButton("Auto Label Objects")
        detection_btn.clicked.connect(lambda: self._request_auto_labeling("Detection"))
        layout.addWidget(detection_btn)
        
        layout.addStretch()
        return widget
        
    def _create_ocr_interface(self) -> QtWidgets.QWidget:
        """Create interface for OCR tasks using standard labelme annotation."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QtWidgets.QLabel("Optical Character Recognition (OCR)")
        title_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # VLM Auto Labeling Section
        vlm_label = QtWidgets.QLabel("VLM Auto OCR:")
        vlm_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(vlm_label)
        
        # Prompt input
        self._ocr_prompt = QtWidgets.QLineEdit()
        self._ocr_prompt.setPlaceholderText("Enter text types to detect (e.g., signs, labels, numbers)")
        layout.addWidget(self._ocr_prompt)
        
        # Auto label button
        ocr_btn = QtWidgets.QPushButton("Auto Detect Text")
        ocr_btn.clicked.connect(lambda: self._request_auto_labeling("OCR"))
        layout.addWidget(ocr_btn)
        
        layout.addStretch()
        return widget
        
    def _create_caption_interface(self) -> QtWidgets.QWidget:
        """Create interface for captioning tasks."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QtWidgets.QLabel("Image & Region Captioning")
        title_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # Prompt history for captioning
        layout.addWidget(QtWidgets.QLabel("Caption Prompt:"))
        self._caption_history = QtWidgets.QListWidget()
        self._caption_history.setMaximumHeight(150)
        self._caption_history.itemDoubleClicked.connect(self._on_edit_prompt)
        self._caption_history.currentRowChanged.connect(self._on_prompt_row_changed)
        layout.addWidget(self._caption_history)
        
        # Add/Remove buttons for captioning (below history)
        caption_btn_layout = QtWidgets.QHBoxLayout()
        add_caption_btn = QtWidgets.QPushButton("Add Prompt")
        add_manual_btn = QtWidgets.QPushButton("Add Manual")
        remove_caption_btn = QtWidgets.QPushButton("Remove Prompt")
        add_caption_btn.clicked.connect(self._add_caption_prompt)
        add_manual_btn.clicked.connect(self._add_manual_caption)
        remove_caption_btn.clicked.connect(self._remove_caption_prompt)
        caption_btn_layout.addWidget(add_caption_btn)
        caption_btn_layout.addWidget(add_manual_btn)
        caption_btn_layout.addWidget(remove_caption_btn)
        layout.addLayout(caption_btn_layout)
        
        # Output area
        layout.addWidget(QtWidgets.QLabel("Output:"))
        self._caption_description = QtWidgets.QTextEdit()
        self._caption_description.setMaximumHeight(300)
        self._caption_description.setPlaceholderText("Caption results and descriptions will appear here...\nEdit this text to modify the output for the selected prompt.")
        self._caption_description.textChanged.connect(self._on_caption_text_changed)
        layout.addWidget(self._caption_description)
        
        layout.addStretch()
        return widget
        
    def _setup_video_interface(self) -> None:
        """Setup the Video category interface (placeholder)."""
        video_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(video_widget)
        
        label = QtWidgets.QLabel("Video Analysis")
        label.setStyleSheet("font-weight: bold; color: #7f8c8d;")
        layout.addWidget(label)
        
        placeholder_label = QtWidgets.QLabel("Video functionality coming soon...")
        placeholder_label.setStyleSheet("color: #95a5a6; font-style: italic;")
        layout.addWidget(placeholder_label)
        
        layout.addStretch()
        self._stacked_widget.addWidget(video_widget)
        
    def _setup_audio_interface(self) -> None:
        """Setup the Audio category interface (placeholder)."""
        audio_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(audio_widget)
        
        label = QtWidgets.QLabel("Audio Analysis")
        label.setStyleSheet("font-weight: bold; color: #7f8c8d;")
        layout.addWidget(label)
        
        placeholder_label = QtWidgets.QLabel("Audio functionality coming soon...")
        placeholder_label.setStyleSheet("color: #95a5a6; font-style: italic;")
        layout.addWidget(placeholder_label)
        
        layout.addStretch()
        self._stacked_widget.addWidget(audio_widget)
        
    def _on_category_changed(self, category: str) -> None:
        """Handle main category change."""
        index_map = {"Image": 0, "Video": 1, "Audio": 2}
        self._stacked_widget.setCurrentIndex(index_map.get(category, 0))
        
    def _on_image_subcategory_changed(self, subcategory: str) -> None:
        """Handle image subcategory change."""
        if self._stacked_widget.currentIndex() == 0:  # Image category
            image_widget = self._stacked_widget.currentWidget()
            index_map = {"Detection": 0, "Caption": 1, "OCR": 2}
            image_widget.subcategory_stack.setCurrentIndex(index_map.get(subcategory, 0))
            # Emit signal to notify main app of subcategory change
            self.subcategory_changed.emit(subcategory)
            
    def _add_caption_prompt(self) -> None:
        """Add new caption prompt through dialog."""
        prompt, ok = self._get_prompt_from_dialog("Add Caption Prompt", "")
        if ok and prompt.strip():
            self._add_to_history(prompt.strip(), "caption")
            # Set the new prompt as current selection
            self._caption_history.setCurrentRow(self._caption_history.count() - 1)
            self.caption_requested.emit(prompt.strip())
    
    def _add_manual_caption(self) -> None:
        """Add manual prompt-output pair without running VLM."""
        prompt, ok = self._get_prompt_from_dialog("Add Caption Prompt", "")
        if ok and prompt.strip():
            # Add to history with empty description for manual editing
            prompt_id = len(self._prompt_history)
            history_item = {
                "prompt": prompt.strip(), 
                "category": "caption", 
                "prompt_id": prompt_id,
                "description": ""
            }
            self._prompt_history.append(history_item)
            
            # Update UI
            self._caption_history.addItem(prompt.strip())
            
            # Select the new item and focus on description area for editing
            self._caption_history.setCurrentRow(self._caption_history.count() - 1)
            # The currentRowChanged signal will automatically trigger the display update
            self._caption_description.setFocus()
            
    def _remove_caption_prompt(self) -> None:
        """Remove selected caption prompt."""
        current_row = self._caption_history.currentRow()
        if current_row >= 0:
            # Remove from UI
            self._caption_history.takeItem(current_row)
            
            # Also remove from internal history
            caption_items = [item for item in self._prompt_history if item.get("category") == "caption"]
            if current_row < len(caption_items):
                self._prompt_history.remove(caption_items[current_row])
            
            # Select previous or clear
            if self._caption_history.count() > 0:
                new_row = min(current_row, self._caption_history.count() - 1)
                self._caption_history.setCurrentRow(new_row)
                # The currentRowChanged signal will automatically trigger the display update
            else:
                self._caption_description.clear()
                
    def _get_prompt_from_dialog(self, title: str, initial: str) -> tuple:
        """
        Helper: open a QDialog with a QPlainTextEdit for multi-line prompts.
        Returns (text, accepted_bool).
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPlainTextEdit, QDialogButtonBox
        
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        layout = QVBoxLayout(dlg)
        edit = QPlainTextEdit()
        edit.setPlainText(initial)
        edit.setMinimumSize(400, 150)
        layout.addWidget(edit)
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        res = dlg.exec_() == QDialog.Accepted
        return edit.toPlainText().strip(), res

    def _on_prompt_row_changed(self, current_row: int) -> None:
        """Handle selection change of caption prompt to display output."""
        if current_row < 0:
            self._caption_description.clear()
            return
            
        # Display the associated description
        caption_items = [item for item in self._prompt_history if item.get("category") == "caption"]
        if current_row < len(caption_items):
            entry = caption_items[current_row]
            # Show the saved description
            desc = entry.get("description", "")
            self._caption_description.setPlainText(desc)
        else:
            self._caption_description.clear()

    def _on_edit_prompt(self, item: QtWidgets.QListWidgetItem) -> None:
        """Double-click handler: edit an existing prompt, re-run inference."""
        current_row = self._caption_history.row(item)
        caption_items = [item for item in self._prompt_history if item.get("category") == "caption"]
        
        if current_row >= len(caption_items):
            return
            
        old_entry = caption_items[current_row]
        old_prompt = old_entry["prompt"]
        
        # Open the same dialog, seeded with the old prompt
        new_prompt, ok = self._get_prompt_from_dialog("Edit Caption Prompt", old_prompt)
        if not ok or not new_prompt.strip() or new_prompt == old_prompt:
            return

        # Update the prompt in history
        old_entry["prompt"] = new_prompt.strip()
        
        # Update UI list
        item.setText(new_prompt.strip())
        self._caption_history.setCurrentRow(current_row)
        
        # Emit signal to re-run VLM inference with new prompt
        self.caption_requested.emit(new_prompt.strip())

    def _add_to_history(self, prompt: str, category: str) -> None:
        """Add prompt to history and update UI."""
        prompt_id = len(self._prompt_history)
        history_item = {"prompt": prompt, "category": category, "prompt_id": prompt_id, "description": ""}
        self._prompt_history.append(history_item)
        
        # Update appropriate history widget
        if category == "caption":
            self._caption_history.addItem(prompt)

    def _update_caption_display_for_prompt(self, caption_index: int) -> None:
        """Update caption display to show output only for selected prompt."""
        self._caption_description.clear()
        
        caption_items = [item for item in self._prompt_history if item.get("category") == "caption"]
        if 0 <= caption_index < len(caption_items):
            description = caption_items[caption_index].get("description", "")
            # Always show the description, even if empty
            self._caption_description.setPlainText(description)
    
    def update_caption_description(self, text: str) -> None:
        """Update caption description area (for new prompts)."""
        # If a specific prompt is selected, update only that prompt's output
        current_row = self._caption_history.currentRow()
        if current_row >= 0:
            caption_items = [item for item in self._prompt_history if item.get("category") == "caption"]
            if current_row < len(caption_items):
                caption_items[current_row]["description"] = text.strip()
                # Update the display directly
                self._caption_description.setPlainText(text.strip())
        else:
            # No specific prompt selected, just append
            self._caption_description.append(text)
    
    def _on_caption_text_changed(self) -> None:
        """Handle changes to caption description text."""
        # Only update if a specific prompt is selected
        current_row = self._caption_history.currentRow()
        if current_row >= 0:
            caption_items = [item for item in self._prompt_history if item.get("category") == "caption"]
            if current_row < len(caption_items):
                new_text = self._caption_description.toPlainText()
                caption_items[current_row]["description"] = new_text
    
    def get_prompt_history(self) -> List[Dict[str, str]]:
        """Get the prompt history."""
        return self._prompt_history.copy()
        
    def set_prompt_history(self, history: List[Dict[str, str]]) -> None:
        """Set the prompt history and update UI."""
        self._prompt_history = []
        
        # Clear existing history
        self._caption_history.clear()
        self._caption_description.clear()
        
        # Populate history widgets based on prompt types (only caption now)
        for item in history:
            prompt = item.get("prompt", "")
            prompt_type = item.get("type", "")
            category = item.get("category", "")
            description = item.get("description", "")
            
            # Only handle caption prompts
            if category == "caption" or prompt_type in ["text", "caption"]:
                # Add to caption category
                self._caption_history.addItem(prompt)
                self._prompt_history.append({
                    "prompt": prompt,
                    "category": "caption", 
                    "description": description,
                    "type": prompt_type
                })
        
        # If there are items, select the first one to show its description
        if self._caption_history.count() > 0:
            self._caption_history.setCurrentRow(0)
            # The currentRowChanged signal will automatically trigger the display update
    
    def _request_auto_labeling(self, task_type: str) -> None:
        """Request auto labeling for Detection or OCR tasks."""
        prompt = ""
        
        if task_type == "Detection":
            prompt = self._detection_prompt.text().strip()
            if not prompt:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Input", "Please enter objects to detect.")
                return
        elif task_type == "OCR":
            prompt = self._ocr_prompt.text().strip()
            if not prompt:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Input", "Please enter text types to detect.")
                return
        
        # Emit signal with prompt and task type
        self.auto_label_requested.emit(prompt, task_type) 