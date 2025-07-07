"""
Utility classes and functions for VLM operations.
"""

import json
import re
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BboxDetection:
    """Represents a detected bounding box with label and confidence."""
    bbox: List[float]  # [x1, y1, x2, y2]
    label: str
    confidence: Optional[float] = None
    description: Optional[str] = None


@dataclass 
class VlmResponse:
    """Standardized response from VLM operations."""
    detections: List[BboxDetection]
    description: str
    raw_output: str


def parse_vlm_json_response(text: str) -> Tuple[List[Dict], str]:
    """
    Parse VLM JSON responses supporting multiple schemas:
    1. [{"bbox": [x1,y1,x2,y2], "label": "...", ...}, ...]
    2. [{"bbox_2d": [x1,y1,x2,y2], "label": "...", ...}, ...]
    3. [{"position": {"x": int, "y": int}, "type": "...", ...}, ...]
    
    Args:
        text: Raw VLM output text
        
    Returns:
        (detections_list, description_text)
        
    Raises:
        ValueError: If JSON parsing fails or format is unrecognized
    """
    # Strip whitespace and extract JSON from code fences if present
    cleaned_text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.+?)```", cleaned_text, re.DOTALL)
    if fence_match:
        cleaned_text = fence_match.group(1).strip()

    try:
        data = json.loads(cleaned_text)
        
        # Schema A: [{"bbox": [...], "label": ...}, ...]
        if (isinstance(data, list) and 
            all(isinstance(item, dict) and "bbox" in item and "label" in item 
                for item in data)):
            return data, ""

        # Schema B: [{"bbox_2d": [...], "label": ...}, ...]  
        if (isinstance(data, list) and
            all(isinstance(item, dict) and "bbox_2d" in item and "label" in item
                for item in data)):
            converted = []
            for item in data:
                converted.append({
                    "bbox": item["bbox_2d"],
                    "label": item["label"],
                    "description": item.get("description", ""),
                    "confidence": item.get("confidence")
                })
            return converted, ""

        # Schema C: [{"position": {"x": ..., "y": ...}, "type": ...}, ...]
        if (isinstance(data, list) and
            all(isinstance(item, dict) and "position" in item and "type" in item
                for item in data)):
            converted = []
            for item in data:
                pos = item["position"]
                x, y = pos.get("x", 0), pos.get("y", 0)
                bbox = [x, y, x, y]  # Point as zero-area bbox
                converted.append({
                    "bbox": bbox,
                    "label": item["type"], 
                    "description": item.get("description", ""),
                    "confidence": item.get("confidence")
                })
            return converted, ""
            
        # If none of the schemas match, raise error
        raise ValueError(f"Unrecognized JSON schema: {data}")
        
    except json.JSONDecodeError as e:
        logger.debug(f"Raw VLM output:\n{text}")
        raise ValueError(f"Failed to parse VLM JSON output: {e}")


def parse_conversation_format_response(text: str) -> Tuple[List[Dict], str]:
    """
    Parse VLM conversation format responses like:
    "There are <p>aircrafts</p>[0.904,0.088,0.994,0.158], <p>aircrafts</p>[0.713,0.535,0.783,0.605] in the image."
    
    Args:
        text: Raw VLM output text in conversation format
        
    Returns:
        (detections_list, description_text)
    """
    # Pattern to match <p>label</p>[x1,y1,x2,y2] format
    pattern = r'<p>([^<]+)</p>\[([0-9.,]+)\]'
    
    matches = re.findall(pattern, text)
    detections = []
    
    for label, coords_str in matches:
        try:
            # Parse coordinates
            coords = [float(x.strip()) for x in coords_str.split(',')]
            
            if len(coords) == 4:  # bbox format
                detection = {
                    "bbox": coords,
                    "label": label.strip(),
                    "description": ""
                }
                detections.append(detection)
            else:
                logger.warning(f"Invalid coordinate format: {coords_str}")
                
        except ValueError as e:
            logger.warning(f"Failed to parse coordinates '{coords_str}': {e}")
            continue
    
    logger.info(f"Parsed {len(detections)} detections from conversation format")
    return detections, text


def is_json_response(text: str) -> bool:
    """Check if VLM response contains JSON data."""
    cleaned = text.strip()
    return (cleaned.startswith('[') and cleaned.endswith(']')) or "```json" in cleaned


def is_conversation_format_response(text: str) -> bool:
    """Check if VLM response contains conversation format data."""
    pattern = r'<p>[^<]+</p>\[[0-9.,]+\]'
    return bool(re.search(pattern, text))


def convert_detections_to_labelme_shapes(
    detections: List[BboxDetection], 
    image_shape: Tuple[int, int],
    model_input_size: Tuple[int, int] = None,
    normalized_coords: bool = True
) -> List[Dict]:
    """
    Convert VLM detections to labelme shape format.
    
    Args:
        detections: List of BboxDetection objects
        image_shape: (height, width) of original image
        model_input_size: (height, width) of model input (optional for normalized coords)
        normalized_coords: Whether coordinates are normalized (0-1) or absolute
        
    Returns:
        List of labelme shape dictionaries
    """
    shapes = []
    img_height, img_width = image_shape
    
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        
        if normalized_coords:
            # Convert normalized coordinates (0-1) to absolute coordinates
            abs_x1 = int(x1 * img_width)
            abs_y1 = int(y1 * img_height)
            abs_x2 = int(x2 * img_width) 
            abs_y2 = int(y2 * img_height)
        else:
            # Scale coordinates from model input to image size
            if model_input_size is None:
                raise ValueError("model_input_size required for non-normalized coordinates")
            input_height, input_width = model_input_size
            abs_x1 = int(x1 / input_width * img_width)
            abs_y1 = int(y1 / input_height * img_height)
            abs_x2 = int(x2 / input_width * img_width) 
            abs_y2 = int(y2 / input_height * img_height)
        
        # Ensure coordinates are in correct order
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
            
        shape = {
            "label": detection.label,
            "points": [[abs_x1, abs_y1], [abs_x2, abs_y2]],
            "group_id": None,
            "shape_type": "rectangle", 
            "flags": {},
            "description": detection.description or ""
        }
        shapes.append(shape)
        
    return shapes


def format_bbox_detection_prompt(objects: str) -> str:
    """
    Format a user prompt for bbox detection into a complete VLM prompt.
    
    Args:
        objects: Comma-separated object names (e.g., "dog,cat,bird")
        
    Returns:
        Complete prompt for VLM bbox detection asking for conversation format
    """
    return (
        f"Find and locate {objects.strip()} in this image. "
        f"Provide the bounding box coordinates and labels in the format: "
        f"There are <p>label</p>[x1,y1,x2,y2], <p>label</p>[x1,y1,x2,y2] in the image. "
        f"Use normalized coordinates between 0 and 1."
    ) 