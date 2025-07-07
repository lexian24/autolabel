"""
Object detection with bounding boxes using VLM.

This module provides functionality for detecting objects and generating
bounding boxes from text prompts using Qwen2.5-VL.
"""

from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from loguru import logger

from .qwen_model import get_model_instance
from .utils import (
    BboxDetection, 
    VlmResponse, 
    parse_vlm_json_response, 
    parse_conversation_format_response,
    convert_detections_to_labelme_shapes,
    format_bbox_detection_prompt,
    is_json_response,
    is_conversation_format_response
)


def detect_objects_with_vlm(
    image: np.ndarray, 
    objects: str
) -> Tuple[List[Dict], str]:
    """
    Detect objects in an image and return bounding boxes.
    
    This is the main function for AI labeling functionality. It takes a list
    of object names and returns detected bounding boxes in labelme format.
    
    Args:
        image: Input image as numpy array (H, W, C)
        objects: Comma-separated object names (e.g., "dog,cat,bird")
        
    Returns:
        (shapes_list, description_text) where:
        - shapes_list: List of labelme shape dictionaries
        - description_text: Any additional description from the model
        
    Raises:
        ValueError: If detection fails or no objects specified
    """
    if not objects or not objects.strip():
        raise ValueError("No objects specified for detection")
        
    logger.debug(f"Detecting objects: {objects}")
    
    # Get model instance
    model = get_model_instance()
    
    # Convert numpy image to PIL
    if image.dtype == np.uint8:
        pil_image = Image.fromarray(image)
    else:
        # Convert to uint8 if needed
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Format the detection prompt
    prompt = format_bbox_detection_prompt(objects)
    logger.debug(f"Using prompt: {prompt}")
    
    # Prepare inputs for coordinate scaling
    inputs = model.processor(
        text=[model.processor.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}, 
                {"image": pil_image}
            ]}
        ], tokenize=False, add_generation_prompt=True)], 
        images=[pil_image], 
        return_tensors="pt"
    ).to(model.device)
    
    # Get model input dimensions for coordinate scaling
    model_input_size = model.get_model_input_size(inputs)
    
    # Run VLM inference
    raw_output = model.inference(image, prompt)
    logger.debug(f"VLM raw output: {raw_output}")
    
    # Check if response contains bounding boxes (JSON or conversation format)
    if is_conversation_format_response(raw_output):
        logger.debug("Parsing conversation format response")
        try:
            # Parse the conversation format response
            detection_dicts, description_text = parse_conversation_format_response(raw_output)
            
            # Convert to BboxDetection objects
            detections = []
            for detection_dict in detection_dicts:
                detection = BboxDetection(
                    bbox=detection_dict["bbox"],
                    label=detection_dict["label"],
                    confidence=detection_dict.get("confidence"),
                    description=detection_dict.get("description", "")
                )
                detections.append(detection)
            
            logger.info(f"Detected {len(detections)} objects from conversation format")
            
            # Convert to labelme shape format (coordinates are normalized)
            image_shape = (image.shape[0], image.shape[1])  # (height, width)
            shapes = convert_detections_to_labelme_shapes(
                detections, 
                image_shape, 
                normalized_coords=True
            )
            
            return shapes, description_text
            
        except Exception as e:
            logger.error(f"Failed to parse conversation format results: {e}")
            raise ValueError(f"Failed to parse conversation format results: {e}")
            
    elif is_json_response(raw_output):
        logger.debug("Parsing JSON format response")
        try:
            # Parse the JSON response
            detection_dicts, description_text = parse_vlm_json_response(raw_output)
            
            # Convert to BboxDetection objects
            detections = []
            for detection_dict in detection_dicts:
                detection = BboxDetection(
                    bbox=detection_dict["bbox"],
                    label=detection_dict["label"],
                    confidence=detection_dict.get("confidence"),
                    description=detection_dict.get("description", "")
                )
                detections.append(detection)
            
            logger.info(f"Detected {len(detections)} objects from JSON format")
            
            # Convert to labelme shape format (coordinates need scaling)
            image_shape = (image.shape[0], image.shape[1])  # (height, width)
            shapes = convert_detections_to_labelme_shapes(
                detections, 
                image_shape, 
                model_input_size,
                normalized_coords=False
            )
            
            return shapes, description_text
            
        except Exception as e:
            logger.error(f"Failed to parse JSON detection results: {e}")
            raise ValueError(f"Failed to parse JSON detection results: {e}")
    else:
        logger.warning("VLM response does not contain recognizable bounding box format")
        return [], raw_output


def validate_detection_objects(objects: str) -> List[str]:
    """
    Validate and parse object names for detection.
    
    Args:
        objects: Comma-separated object names
        
    Returns:
        List of cleaned object names
        
    Raises:
        ValueError: If no valid objects found
    """
    if not objects or not objects.strip():
        raise ValueError("No objects specified")
        
    # Split by comma and clean up
    object_list = [obj.strip() for obj in objects.split(",")]
    object_list = [obj for obj in object_list if obj]  # Remove empty strings
    
    if not object_list:
        raise ValueError("No valid objects found after parsing")
        
    logger.debug(f"Parsed objects: {object_list}")
    return object_list


def get_detection_confidence_threshold() -> float:
    """
    Get the default confidence threshold for detections.
    
    Returns:
        Default confidence threshold (0.3)
    """
    return 0.3


def filter_detections_by_confidence(
    detections: List[BboxDetection], 
    threshold: float = None
) -> List[BboxDetection]:
    """
    Filter detections by confidence score.
    
    Args:
        detections: List of BboxDetection objects
        threshold: Confidence threshold (default from get_detection_confidence_threshold)
        
    Returns:
        Filtered list of detections
    """
    if threshold is None:
        threshold = get_detection_confidence_threshold()
        
    filtered = []
    for detection in detections:
        if detection.confidence is None or detection.confidence >= threshold:
            filtered.append(detection)
        else:
            logger.debug(f"Filtered out detection {detection.label} "
                        f"with confidence {detection.confidence}")
            
    logger.info(f"Filtered {len(detections)} → {len(filtered)} detections "
               f"(threshold: {threshold})")
    return filtered 