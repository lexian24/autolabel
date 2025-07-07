"""
Image description and analysis using VLM.

This module provides functionality for generating text descriptions
of images and specific regions using Qwen2.5-VL.
"""

import tempfile
import os
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw
from PyQt5 import QtGui
from loguru import logger

from .qwen_model import get_model_instance


def get_image_description(
    image_path: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 4096
) -> str:
    """
    Generate a text description of an image based on a custom prompt.
    
    This is the main function for custom VLM prompting functionality.
    
    Args:
        image_path: Path to the image file
        prompt: Custom text prompt for the VLM
        system_prompt: System message for the model
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text description from the VLM
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If inference fails
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    logger.debug(f"Generating description for image: {image_path}")
    logger.debug(f"Using prompt: {prompt}")
    
    # Get model instance and run inference
    model = get_model_instance()
    
    try:
        description = model.inference(
            image=image_path,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens
        )
        
        logger.info(f"Generated description ({len(description)} chars)")
        return description.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate image description: {e}")
        raise ValueError(f"VLM inference failed: {e}")


def describe_bbox_region(
    qimage: QtGui.QImage,
    bbox_coords: tuple[int, int, int, int],
    prompt: str = "What is in the bounding box?",
    highlight_color: tuple[int, int, int] = (255, 0, 0),
    highlight_width: int = 4
) -> str:
    """
    Generate a description of a specific bounding box region in an image.
    
    This function is used for the right-click "Describe" feature on bounding boxes.
    It highlights the selected region and asks the VLM to describe its contents.
    
    Args:
        qimage: Qt image object
        bbox_coords: Bounding box coordinates (x1, y1, x2, y2)
        prompt: Question prompt for the VLM
        highlight_color: RGB color for highlighting the bbox
        highlight_width: Width of the highlight border
        
    Returns:
        VLM-generated description of the bbox region
        
    Raises:
        ValueError: If bbox coordinates are invalid or inference fails
    """
    x1, y1, x2, y2 = bbox_coords
    
    # Validate bbox coordinates
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid bbox coordinates: ({x1}, {y1}, {x2}, {y2})")
        
    logger.debug(f"Describing bbox region: ({x1}, {y1}, {x2}, {y2})")
    
    # Create a copy of the image and draw the highlight
    img_copy = QtGui.QImage(qimage)
    painter = QtGui.QPainter(img_copy)
    
    # Set up the highlight pen
    pen = QtGui.QPen(QtGui.QColor(*highlight_color))
    pen.setWidth(highlight_width)
    painter.setPen(pen)
    
    # Draw the highlight rectangle
    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
    painter.end()
    
    # Save to temporary file for VLM processing
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    
    try:
        # Save the highlighted image
        img_copy.save(tmp_path, "PNG")
        
        # Generate description using VLM
        description = get_image_description(tmp_path, prompt)
        
        logger.info(f"Generated bbox description ({len(description)} chars)")
        return description
        
    except Exception as e:
        logger.error(f"Failed to describe bbox region: {e}")
        raise ValueError(f"Bbox description failed: {e}")
        
    finally:
        # Clean up temporary file
        try:
            os.remove(tmp_path)
        except OSError as e:
            logger.warning(f"Failed to remove temporary file {tmp_path}: {e}")


def create_highlighted_image(
    image: np.ndarray,
    bbox_coords: tuple[int, int, int, int],
    highlight_color: tuple[int, int, int] = (255, 0, 0),
    highlight_width: int = 4
) -> np.ndarray:
    """
    Create an image with a highlighted bounding box region.
    
    Args:
        image: Input image as numpy array
        bbox_coords: Bounding box coordinates (x1, y1, x2, y2)
        highlight_color: RGB color for highlighting
        highlight_width: Width of the highlight border
        
    Returns:
        Image with highlighted bbox as numpy array
    """
    x1, y1, x2, y2 = bbox_coords
    
    # Convert to PIL Image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    
    # Create drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Draw the highlight rectangle
    for i in range(highlight_width):
        draw.rectangle(
            [x1 - i, y1 - i, x2 + i, y2 + i],
            outline=highlight_color,
            width=1
        )
    
    # Convert back to numpy array
    return np.array(pil_image)


def validate_prompt(prompt: str) -> str:
    """
    Validate and clean a user prompt.
    
    Args:
        prompt: User input prompt
        
    Returns:
        Cleaned prompt text
        
    Raises:
        ValueError: If prompt is empty or invalid
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
        
    cleaned = prompt.strip()
    
    # Basic length validation
    if len(cleaned) > 2000:
        logger.warning(f"Prompt is very long ({len(cleaned)} chars), truncating...")
        cleaned = cleaned[:2000] + "..."
        
    return cleaned


def format_describe_prompt(base_prompt: str = "What is in the bounding box?") -> str:
    """
    Format a prompt for bbox description with helpful context.
    
    Args:
        base_prompt: Base question to ask about the bbox
        
    Returns:
        Formatted prompt with additional context
    """
    return (
        f"{base_prompt.strip()} "
        f"Please provide a detailed description of the objects and their characteristics "
        f"within the highlighted red bounding box."
    ) 