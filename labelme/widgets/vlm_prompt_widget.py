"""
VLM Prompt Widget.

This widget provides a UI for custom VLM prompting functionality.
Users can enter custom prompts to ask questions about images.
"""

from loguru import logger
from PyQt5 import QtWidgets
from typing import Callable


class VlmPromptWidget(QtWidgets.QWidget):
    """
    Widget for custom VLM prompting and image description.
    
    This widget replaces the old AiPromptWidget with a clearer name and purpose.
    It provides a text input for custom prompts and a submit button.
    """
    
    def __init__(self, on_submit_callback: Callable[[], None], parent=None):
        """
        Initialize the VLM prompt widget.
        
        Args:
            on_submit_callback: Function to call when prompt is submitted
            parent: Parent Qt widget
        """
        super().__init__(parent=parent)
        self._setup_ui(on_submit_callback)

    def _setup_ui(self, on_submit_callback: Callable[[], None]) -> None:
        """Set up the user interface components."""
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(0)  # type: ignore[union-attr]

        # Create the main input widget
        input_widget = _PromptInputWidget(
            on_submit_callback=on_submit_callback, 
            parent=self
        )
        input_widget.setMaximumWidth(400)
        self.layout().addWidget(input_widget)  # type: ignore[union-attr]

    def get_prompt_text(self) -> str:
        """
        Get the current prompt text from the input field.
        
        Returns:
            User-entered prompt text
        """
        layout = self.layout()
        if not layout:
            logger.warning("Widget layout not found")
            return ""
            
        item = layout.itemAt(0)
        if not item or not item.widget():
            logger.warning("Input widget not found")
            return ""
            
        return item.widget().get_prompt_text()
    
    def set_prompt_text(self, prompt: str) -> None:
        """
        Set the prompt text in the input field.
        
        Args:
            prompt: Prompt text to populate
        """
        layout = self.layout()
        if not layout:
            return
            
        item = layout.itemAt(0)
        if not item or not item.widget():
            return
            
        item.widget().set_prompt_text(prompt)
        
    def clear_input(self) -> None:
        """Clear the prompt input field."""
        self.set_prompt_text("")


class _PromptInputWidget(QtWidgets.QWidget):
    """
    Internal widget for prompt input controls.
    
    Contains label, text input, and submit button in horizontal layout.
    """
    
    def __init__(self, on_submit_callback: Callable[[], None], parent=None):
        """
        Initialize the input widget.
        
        Args:
            on_submit_callback: Function to call when prompt is submitted
            parent: Parent Qt widget
        """
        super().__init__(parent=parent)
        self._setup_ui(on_submit_callback)

    def _setup_ui(self, on_submit_callback: Callable[[], None]) -> None:
        """Set up the horizontal layout with controls."""
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)  # type: ignore[union-attr]

        # Label
        label = QtWidgets.QLabel(self.tr("VLM Prompt"))
        label.setToolTip(self.tr("Enter a custom prompt for image analysis"))
        self.layout().addWidget(label)  # type: ignore[union-attr]

        # Text input field
        self._text_input = QtWidgets.QLineEdit()
        self._text_input.setPlaceholderText(
            self.tr("e.g., Describe the image in deatil.")
        )
        self._text_input.setToolTip(
            self.tr("Enter a custom question or instruction for the VLM")
        )
        
        # Connect Enter key to submission
        self._text_input.returnPressed.connect(on_submit_callback)
        self.layout().addWidget(self._text_input)  # type: ignore[union-attr]

        # Submit button
        submit_button = QtWidgets.QPushButton(text=self.tr("Ask VLM"), parent=self)
        submit_button.setToolTip(self.tr("Submit the prompt to the Vision-Language Model"))
        submit_button.clicked.connect(on_submit_callback)
        self.layout().addWidget(submit_button)  # type: ignore[union-attr]

    def get_prompt_text(self) -> str:
        """Get the current text from the input field."""
        return self._text_input.text().strip()

    def set_prompt_text(self, prompt: str) -> None:
        """Set the text in the input field."""
        self._text_input.setText(prompt)
        
    def focus_input(self) -> None:
        """Set focus to the text input field."""
        self._text_input.setFocus()
        
    def select_all_text(self) -> None:
        """Select all text in the input field."""
        self._text_input.selectAll() 