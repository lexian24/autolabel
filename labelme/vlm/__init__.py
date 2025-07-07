"""
Vision-Language Model (VLM) integration for labelme.

This module provides Qwen2.5-VL integration for:
- Object detection with bounding boxes
- Image description and analysis
- Interactive prompt-based querying
"""

from .qwen_model import QwenVLModel
from .bbox_detection import detect_objects_with_vlm
from .description import get_image_description, describe_bbox_region
from .utils import VlmResponse, BboxDetection

__all__ = [
    'QwenVLModel',
    'detect_objects_with_vlm', 
    'get_image_description',
    'describe_bbox_region',
    'VlmResponse',
    'BboxDetection'
] 