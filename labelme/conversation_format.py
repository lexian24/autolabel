"""
Conversation Format Support for Labelme.

Handles loading and exporting conversation-based annotation format.
Supports both grounding conversations (with coordinates) and pure text conversations.
"""

import json
import re
import os.path as osp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from labelme import utils
from labelme.label_file import LabelFile


@dataclass
class ConversationAnnotation:
    """Represents a single annotation from a conversation."""
    label: str
    coordinates: List[float]  # Normalized coordinates [0-1]
    annotation_type: str  # 'bbox', 'polygon', 'point'


@dataclass
class ConversationEntry:
    """Represents a single conversation turn."""
    speaker: str  # 'human' or 'gpt'
    content: str
    annotations: List[ConversationAnnotation]  # Only for GPT responses
    has_grounding: bool = False  # Whether this entry contains spatial annotations


class ConversationFormatLoader:
    """Loads conversation format files and converts to labelme format."""
    
    def __init__(self):
        self.annotation_pattern = re.compile(r'<p>([^<]+)</p>\[([^\]]+)\]')
    
    def load(self, filename: str) -> Optional[LabelFile]:
        """Load a conversation format file."""
        try:
            logger.info(f"Loading conversation format file: {filename}")
            
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"Successfully parsed JSON, keys: {list(data.keys())}")
            
            if 'conversations' in data:
                logger.info(f"Found {len(data['conversations'])} conversations")
            
            result = self._create_label_file_from_data(data, filename)
            
            if result:
                logger.info(f"Successfully created LabelFile with {len(result.shapes)} shapes")
            else:
                logger.error(f"Failed to create LabelFile from data")
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to load conversation format file {filename}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def analyze_conversations(self, conversations: List[Dict]) -> Dict[str, int]:
        """Analyze conversation content to categorize types."""
        stats = {
            'total_conversations': len(conversations),
            'grounding_conversations': 0,  # Contains coordinates/annotations
            'pure_text_conversations': 0,  # Pure text descriptions
            'total_annotations': 0
        }
        
        for conv in conversations:
            if conv.get('from') == 'gpt':
                annotations = self._parse_gpt_annotations(conv.get('value', ''))
                if annotations:
                    stats['grounding_conversations'] += 1
                    stats['total_annotations'] += len(annotations)
                else:
                    stats['pure_text_conversations'] += 1
                    
        return stats
    
    def separate_conversation_types(self, conversations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Separate grounding conversations from pure text conversations."""
        grounding_conversations = []
        text_conversations = []
        
        i = 0
        while i < len(conversations):
            # Process human-gpt pairs
            if i + 1 < len(conversations):
                human_conv = conversations[i]
                gpt_conv = conversations[i + 1]
                
                if (human_conv.get('from') == 'human' and 
                    gpt_conv.get('from') == 'gpt'):
                    
                    # Check if GPT response has grounding annotations
                    annotations = self._parse_gpt_annotations(gpt_conv.get('value', ''))
                    
                    conversation_pair = [human_conv, gpt_conv]
                    
                    if annotations:
                        grounding_conversations.extend(conversation_pair)
                    else:
                        text_conversations.extend(conversation_pair)
                    
                    i += 2  # Skip both human and gpt
                else:
                    # Single conversation entry
                    text_conversations.append(conversations[i])
                    i += 1
            else:
                # Single conversation entry
                text_conversations.append(conversations[i])
                i += 1
                
        return grounding_conversations, text_conversations
    
    def _create_label_file_from_data(self, data: Dict, filename: str) -> Optional[LabelFile]:
        """Create a LabelFile object from conversation data."""
        if 'image' not in data:
            return None
            
        image_path = data['image']
        
        # Convert relative path to absolute if needed
        if not osp.isabs(image_path):
            base_dir = osp.dirname(filename)
            image_path = osp.join(base_dir, image_path)
            
        logger.info(f"Looking for image at: {image_path}")
        
        if not osp.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            logger.error(f"Base directory: {osp.dirname(filename)}")
            logger.error(f"Original image path in JSON: {data['image']}")
            
            # Try alternative paths
            base_dir = osp.dirname(filename)
            alt_image_path = osp.join(base_dir, osp.basename(data['image']))
            if osp.exists(alt_image_path):
                logger.info(f"Found image at alternative path: {alt_image_path}")
                image_path = alt_image_path
            else:
                logger.error(f"Alternative path also doesn't exist: {alt_image_path}")
                return None
            
        # Load image data
        try:
            image_data = LabelFile.load_image_file(image_path)
            if not image_data:
                return None
                
            # Get image dimensions
            import PIL.Image
            import io
            img_pil = PIL.Image.open(io.BytesIO(image_data))
            img_width, img_height = img_pil.size
            
            # Analyze conversations
            conversations = data.get('conversations', [])
            stats = self.analyze_conversations(conversations)
            grounding_convs, text_convs = self.separate_conversation_types(conversations)
            
            logger.info(f"Conversation analysis: {stats}")
            logger.info(f"Grounding conversations: {len(grounding_convs)//2}, Text conversations: {len(text_convs)//2}")
            
            # Convert grounding conversation annotations to labelme shapes
            shapes = []
            for conv in conversations:
                if conv.get('from') == 'gpt':
                    annotations = self._parse_gpt_annotations(conv.get('value', ''))
                    for ann in annotations:
                        shape = self._convert_annotation_to_shape(ann, img_width, img_height)
                        if shape:
                            shapes.append(shape)
            
            # Create LabelFile object
            label_file = LabelFile()
            label_file.shapes = shapes
            label_file.imagePath = image_path  # Store full absolute path
            label_file.imageData = image_data
            label_file.filename = filename
            label_file.flags = {}
            
            # Reconstruct prompt history from conversations
            prompt_history = self._reconstruct_prompt_history(conversations)
            
            # Store conversation data and analysis in otherData
            label_file.otherData = {
                'conversations': conversations,
                'grounding_conversations': grounding_convs,
                'text_conversations': text_convs,
                'conversation_stats': stats,
                'prompt_history': prompt_history,
                'vlm_description': prompt_history[-1].get('description', '') if prompt_history else ''
            }
            
            # Log unique labels found
            unique_labels = set(shape['label'] for shape in shapes)
            logger.info(f"Loaded {len(shapes)} shapes with labels: {sorted(unique_labels)}")
            
            return label_file
            
        except Exception as e:
            logger.error(f"Failed to create LabelFile from conversation data: {e}")
            return None
    
    def _reconstruct_prompt_history(self, conversations: List[Dict]) -> List[Dict]:
        """Reconstruct prompt history from conversations for UI display."""
        prompt_history = []
        
        i = 0
        while i < len(conversations):
            # Process human-gpt pairs
            if (i + 1 < len(conversations) and 
                conversations[i].get('from') == 'human' and 
                conversations[i + 1].get('from') == 'gpt'):
                
                human_conv = conversations[i]
                gpt_conv = conversations[i + 1]
                
                prompt = human_conv.get('value', '')
                description = gpt_conv.get('value', '')
                attribute = human_conv.get('attribute', None)  # May not exist in older files
                
                # Determine entry type and category based on attribute or content analysis
                if attribute == 'Grounding':
                    # Check if it has detected objects or is AI labeling
                    annotations = self._parse_gpt_annotations(description)
                    if annotations:
                        # Determine if it's AI labeling or object detection based on prompt
                        if any(phrase in prompt.lower() for phrase in ['detect and locate', 'find and locate']):
                            entry_type = 'ai_labeling'
                        else:
                            entry_type = 'object_detection'
                        
                        detected_objects = [ann.label for ann in annotations]
                        entry = {
                            'prompt': prompt,
                            'description': description,
                            'type': entry_type,
                            'category': 'grounding',
                            'detected_objects': detected_objects,
                            'detection_count': len(annotations)
                        }
                    else:
                        entry = {
                            'prompt': prompt,
                            'description': description,
                            'type': 'grounding',
                            'category': 'grounding'
                        }
                        
                elif attribute == 'Region Captioning':
                    entry = {
                        'prompt': prompt,
                        'description': description,
                        'type': 'bbox_description',
                        'category': 'grounding'  # Region captioning is a type of grounding
                    }
                    
                elif attribute == 'Image Captioning':
                    entry = {
                        'prompt': prompt,
                        'description': description,
                        'type': 'text',
                        'category': 'caption'
                    }
                    
                else:
                    # No attribute field - analyze content to determine type (backward compatibility)
                    annotations = self._parse_gpt_annotations(description)
                    if annotations:
                        # Has annotations - it's grounding
                        if any(phrase in prompt.lower() for phrase in ['detect and locate', 'find and locate']):
                            entry_type = 'ai_labeling'
                        elif any(phrase in prompt.lower() for phrase in ['detect', 'find', 'locate']):
                            entry_type = 'object_detection'
                        else:
                            entry_type = 'grounding'
                            
                        detected_objects = [ann.label for ann in annotations]
                        entry = {
                            'prompt': prompt,
                            'description': description,
                            'type': entry_type,
                            'category': 'grounding',
                            'detected_objects': detected_objects,
                            'detection_count': len(annotations)
                        }
                    else:
                        # No annotations - check if it's descriptive
                        if any(phrase in prompt.lower() for phrase in ['describe', 'what do you see', 'what is in']):
                            entry_type = 'text'
                        else:
                            entry_type = 'text'  # Default fallback
                            
                        entry = {
                            'prompt': prompt,
                            'description': description,
                            'type': entry_type,
                            'category': 'caption'
                        }
                
                prompt_history.append(entry)
                i += 2  # Skip both human and gpt
            else:
                # Single conversation - skip or handle specially
                i += 1
        
        return prompt_history
    
    def _parse_gpt_annotations(self, gpt_response: str) -> List[ConversationAnnotation]:
        """Parse annotations from GPT response text."""
        annotations = []
        
        matches = self.annotation_pattern.findall(gpt_response)
        for label, coord_str in matches:
            try:
                # Parse coordinates
                coords = [float(x.strip()) for x in coord_str.split(',')]
                
                # Determine annotation type based on coordinate count
                if len(coords) == 2:
                    ann_type = 'point'
                elif len(coords) == 4:
                    ann_type = 'bbox'
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    ann_type = 'polygon'
                else:
                    logger.warning(f"Unrecognized coordinate format: {coords}")
                    continue
                
                annotations.append(ConversationAnnotation(
                    label=label.strip(),
                    coordinates=coords,
                    annotation_type=ann_type
                ))
                
            except ValueError as e:
                logger.warning(f"Failed to parse coordinates '{coord_str}': {e}")
                continue
                
        return annotations
    
    def _convert_annotation_to_shape(self, annotation: ConversationAnnotation, 
                                   img_width: int, img_height: int) -> Optional[Dict]:
        """Convert conversation annotation to labelme shape format."""
        try:
            coords = annotation.coordinates
            
            if annotation.annotation_type == 'point':
                # Point: [x, y] normalized -> [[x_abs, y_abs]]
                x_abs = coords[0] * img_width
                y_abs = coords[1] * img_height
                points = [[x_abs, y_abs]]
                shape_type = 'point'
                
            elif annotation.annotation_type == 'bbox':
                # Bbox: [x1, y1, x2, y2] normalized -> [[x1, y1], [x2, y2]]
                x1_abs = coords[0] * img_width
                y1_abs = coords[1] * img_height
                x2_abs = coords[2] * img_width
                y2_abs = coords[3] * img_height
                points = [[x1_abs, y1_abs], [x2_abs, y2_abs]]
                shape_type = 'rectangle'
                
            elif annotation.annotation_type == 'polygon':
                # Polygon: [x1, y1, x2, y2, ...] normalized -> [[x1, y1], [x2, y2], ...]
                points = []
                for i in range(0, len(coords), 2):
                    x_abs = coords[i] * img_width
                    y_abs = coords[i + 1] * img_height
                    points.append([x_abs, y_abs])
                shape_type = 'polygon'
                
            else:
                logger.warning(f"Unknown annotation type: {annotation.annotation_type}")
                return None
            
            return {
                'label': annotation.label,
                'points': points,
                'group_id': None,
                'shape_type': shape_type,
                'flags': {},
                'description': '',
                'mask': None,
                'other_data': {}
            }
            
        except Exception as e:
            logger.error(f"Failed to convert annotation to shape: {e}")
            return None


class ConversationFormatExporter:
    """Exports labelme format to conversation format."""
    
    def export(self, label_file: LabelFile, output_filename: str, 
               include_text_conversations: bool = True) -> bool:
        """Export labelme data to conversation format."""
        try:
            # Get image dimensions
            image_data = label_file.imageData
            if not image_data:
                logger.error(f"No image data available for export. LabelFile path: {label_file.imagePath}")
                logger.error(f"LabelFile filename: {getattr(label_file, 'filename', 'N/A')}")
                logger.error(f"Image data length: {len(image_data) if image_data else 'None'}")
                return False
                
            import PIL.Image
            import io
            img_pil = PIL.Image.open(io.BytesIO(image_data))
            img_width, img_height = img_pil.size
            
            # Get current data (always regenerate conversations based on current state)
            prompt_history = label_file.otherData.get('prompt_history', [])
            
            logger.info(f"Regenerating conversations from current state:")
            logger.info(f"  - Shapes: {len(label_file.shapes)}")
            logger.info(f"  - Prompt history: {len(prompt_history)}")
            
            # Always generate new conversations based on current state
            conversations = []
            
            # Add grounding conversation based on ALL current shapes (comprehensive labeling)
            if label_file.shapes:
                logger.info(f"Generating grounding conversation for {len(label_file.shapes)} shapes")
                grounding_conv = self._generate_grounding_conversation(
                    label_file.shapes, img_width, img_height
                )
                conversations.extend(grounding_conv)
                
                # Log the unique labels being exported
                unique_labels = set(shape['label'] for shape in label_file.shapes)
                logger.info(f"  - Unique labels: {sorted(unique_labels)}")
            
            # Add AI labeling prompts as separate conversation entries
            if prompt_history:
                logger.info(f"Generating AI prompt conversations for {len(prompt_history)} entries")
                ai_convs = self._generate_ai_prompt_conversations(prompt_history, img_width, img_height)
                conversations.extend(ai_convs)
                
                # Log prompt types
                types = [entry.get('type', 'text') for entry in prompt_history]
                logger.info(f"  - Prompt types: {types}")
            
            # Add text conversations if they exist and are requested
            if include_text_conversations:
                vlm_description = label_file.otherData.get('vlm_description', '')
                if vlm_description.strip():
                    logger.info("Adding VLM description as text conversation")
                    text_conv = self._generate_text_conversation(vlm_description)
                    conversations.extend(text_conv)
            
            logger.info(f"Total conversations generated: {len(conversations)}")
            
            # Create conversation format data
            conv_data = {
                'image': label_file.imagePath,
                'conversations': conversations
            }
            
            # Write to file
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(conv_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Exported conversation format to: {output_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export conversation format: {e}")
            return False
    
    def _generate_grounding_conversation(self, shapes: List[Dict], 
                                       img_width: int, img_height: int) -> List[Dict]:
        """Generate grounding conversation from labelme shapes."""
        if not shapes:
            return []
        
        # Generate human prompt based on shapes
        labels = list(set(shape['label'] for shape in shapes))
        shape_types = list(set(shape['shape_type'] for shape in shapes))
        
        if 'rectangle' in shape_types or 'polygon' in shape_types:
            annotation_type = "bounding boxes" if 'rectangle' in shape_types else "polygons"
        elif 'point' in shape_types:
            annotation_type = "points"
        else:
            annotation_type = "annotations"
        
        if len(labels) == 1:
            prompt = f"Detect all {labels[0]} in the image and describe using {annotation_type}."
        else:
            prompt = f"Detect all objects in the image and describe using {annotation_type}."
        
        # Generate GPT response with annotations
        gpt_response = self._format_annotations_in_response(shapes, img_width, img_height)
        
        return [
            {
                'from': 'human',
                'value': prompt,
                'attribute': 'Grounding'
            },
            {
                'from': 'gpt',
                'value': gpt_response,
                'attribute': 'Grounding'
            }
        ]
    
    def _generate_text_conversation(self, description: str) -> List[Dict]:
        """Generate pure text conversation from description."""
        return [
            {
                'from': 'human',
                'value': 'Describe what you see in the image.',
                'attribute': 'Image Captioning'
            },
            {
                'from': 'gpt',
                'value': description,
                'attribute': 'Image Captioning'
            }
        ]
    
    def _generate_ai_prompt_conversations(self, prompt_history: List[Dict], 
                                        img_width: int, img_height: int) -> List[Dict]:
        """Generate conversation entries from AI prompt history."""
        conversations = []
        
        for entry in prompt_history:
            prompt = entry.get('prompt', '')
            description = entry.get('description', '')
            entry_type = entry.get('type', 'text')
            
            if not prompt or not description:
                continue
                
            # Skip pure text prompts if they're not AI labeling related
            if entry_type in ['ai_labeling', 'object_detection']:
                # AI labeling/detection prompts should be included as separate conversations
                conversations.extend([
                    {
                        'from': 'human',
                        'value': prompt,
                        'attribute': 'Grounding'
                    },
                    {
                        'from': 'gpt',
                        'value': description,
                        'attribute': 'Grounding'
                    }
                ])
            elif entry_type == 'bbox_description':
                # Region captioning for bbox descriptions
                conversations.extend([
                    {
                        'from': 'human',
                        'value': prompt,
                        'attribute': 'Region Captioning'
                    },
                    {
                        'from': 'gpt',
                        'value': description,
                        'attribute': 'Region Captioning'
                    }
                ])
            elif entry_type == 'text':
                # Image captioning for general descriptions
                conversations.extend([
                    {
                        'from': 'human',
                        'value': prompt,
                        'attribute': 'Image Captioning'
                    },
                    {
                        'from': 'gpt',
                        'value': description,
                        'attribute': 'Image Captioning'
                    }
                ])
            elif 'bbox' in entry or entry.get('detected_objects'):
                # Include any prompt that resulted in detections or has bbox info
                conversations.extend([
                    {
                        'from': 'human',
                        'value': prompt,
                        'attribute': 'Grounding'
                    },
                    {
                        'from': 'gpt',
                        'value': description,
                        'attribute': 'Grounding'
                    }
                ])
        
        return conversations
    
    def _format_annotations_in_response(self, shapes: List[Dict], 
                                      img_width: int, img_height: int) -> str:
        """Format labelme shapes as conversation response text."""
        if not shapes:
            return "I don't see any specific objects to detect in this image."
            
        parts = []
        
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            shape_type = shape['shape_type']
            
            # Convert to normalized coordinates
            if shape_type == 'point':
                # Point: [[x, y]] -> [x_norm, y_norm]
                x_norm = points[0][0] / img_width
                y_norm = points[0][1] / img_height
                coord_str = f"{x_norm:.3f},{y_norm:.3f}"
                
            elif shape_type == 'rectangle':
                # Rectangle: [[x1, y1], [x2, y2]] -> [x1_norm, y1_norm, x2_norm, y2_norm]
                x1_norm = points[0][0] / img_width
                y1_norm = points[0][1] / img_height
                x2_norm = points[1][0] / img_width
                y2_norm = points[1][1] / img_height
                coord_str = f"{x1_norm:.3f},{y1_norm:.3f},{x2_norm:.3f},{y2_norm:.3f}"
                
            elif shape_type == 'polygon':
                # Polygon: [[x1, y1], [x2, y2], ...] -> [x1_norm, y1_norm, x2_norm, y2_norm, ...]
                coords = []
                for point in points:
                    x_norm = point[0] / img_width
                    y_norm = point[1] / img_height
                    coords.extend([f"{x_norm:.3f}", f"{y_norm:.3f}"])
                coord_str = ",".join(coords)
                
            else:
                logger.warning(f"Unsupported shape type for export: {shape_type}")
                continue
                
            parts.append(f"<p>{label}</p>[{coord_str}]")
        
        if len(parts) == 1:
            return f"There is {parts[0]} in the image."
        elif len(parts) == 2:
            return f"There are {parts[0]} and {parts[1]} in the image."
        else:
            formatted_parts = ", ".join(parts[:-1]) + f", and {parts[-1]}"
            return f"There are {formatted_parts} in the image."


class ShareGPTExporter:
    """Exports labelme format to ShareGPT format."""
    
    def export_single_file(self, label_file: LabelFile, output_filename: str) -> bool:
        """
        Export labelme data to ShareGPT format as a single file with all conversations concatenated.
        Returns True if successful.
        """
        try:
            # Get image dimensions
            image_data = label_file.imageData
            if not image_data:
                logger.error(f"No image data available for export. LabelFile path: {label_file.imagePath}")
                return False
                
            import PIL.Image
            import io
            img_pil = PIL.Image.open(io.BytesIO(image_data))
            img_width, img_height = img_pil.size
            
            # Get current data - caption_history can be in otherData or at root level
            caption_history = []
            if hasattr(label_file, 'otherData') and label_file.otherData:
                caption_history = label_file.otherData.get('caption_history', [])
            
            logger.info(f"Exporting ShareGPT format to single file:")
            logger.info(f"  - Shapes: {len(label_file.shapes)}")
            logger.info(f"  - Caption history: {len(caption_history)}")
            
            conversations = []
            
            # Add caption conversations (one per prompt-output pair)
            for entry in caption_history:
                prompt = entry.get('prompt', '').strip()
                description = entry.get('description', '').strip()
                
                if prompt and description:
                    conversations.extend([
                        {
                            "from": "human",
                            "value": prompt
                        },
                        {
                            "from": "gpt", 
                            "value": description
                        }
                    ])
            
            # Group shapes by task type for grounding conversations
            detection_shapes = []
            ocr_shapes = []
            
            for shape in label_file.shapes:
                # vlm_task is a direct property on shapes, not in other_data
                task_type = shape.get('vlm_task')
                
                if task_type == 'Detection':
                    detection_shapes.append(shape)
                elif task_type == 'OCR':
                    ocr_shapes.append(shape)
                # Only add shapes that have explicit task types
                # Don't default untagged shapes to any category
            
            # Create separate JSON objects for each type
            json_objects = []
            
            # Add caption conversations (one per prompt-output pair)
            for entry in caption_history:
                prompt = entry.get('prompt', '').strip()
                description = entry.get('description', '').strip()
                
                if prompt and description:
                    json_objects.append({
                        "image": osp.basename(label_file.imagePath or "image.jpg"),
                        "task": "Caption",
                        "conversations": [
                            {
                                "from": "human",
                                "value": prompt
                            },
                            {
                                "from": "gpt", 
                                "value": description
                            }
                        ]
                    })
            
            # Add detection conversation with new prompt
            if detection_shapes:
                human_prompt = "Outline all object with bounding box using this format: <p>label</p>[x1,y1,x2,y2], <p>label</p>[x1,y1,x2,y2]"
                gpt_response = self._format_shapes_as_json(detection_shapes, img_width, img_height)
                
                json_objects.append({
                    "image": osp.basename(label_file.imagePath or "image.jpg"),
                    "task": "Detection",
                    "conversations": [
                        {
                            "from": "human",
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": gpt_response
                        }
                    ]
                })
            
            # Add OCR conversation with new prompt
            if ocr_shapes:
                human_prompt = "Read and Outline all words with bounding box using this format: <p>label</p>[x1,y1,x2,y2], <p>label</p>[x1,y1,x2,y2]"
                gpt_response = self._format_shapes_as_json(ocr_shapes, img_width, img_height)
                
                json_objects.append({
                    "image": osp.basename(label_file.imagePath or "image.jpg"),
                    "task": "OCR",
                    "conversations": [
                        {
                            "from": "human", 
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": gpt_response
                        }
                    ]
                })
            
            # If no conversations, return False
            if not json_objects:
                logger.warning("No conversations to export")
                return False
            
            # Write all JSON objects to file (one per line or separated)
            with open(output_filename, 'w', encoding='utf-8') as f:
                for i, obj in enumerate(json_objects):
                    if i > 0:
                        f.write('\n')  # Separate objects with newline
                    json.dump(obj, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully exported ShareGPT format to: {output_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export ShareGPT format: {e}")
            return False
    
    def export_to_directory(self, label_file: LabelFile, output_dir: str) -> int:
        """
        Export labelme data to ShareGPT format as separate files per prompt-output pair.
        Returns the number of files created.
        """
        try:
            import os
            
            # Get image dimensions
            image_data = label_file.imageData
            if not image_data:
                logger.error(f"No image data available for export. LabelFile path: {label_file.imagePath}")
                return 0
                
            import PIL.Image
            import io
            img_pil = PIL.Image.open(io.BytesIO(image_data))
            img_width, img_height = img_pil.size
            
            # Get current data - caption_history can be in otherData or at root level  
            caption_history = []
            if hasattr(label_file, 'otherData') and label_file.otherData:
                caption_history = label_file.otherData.get('caption_history', [])
            
            logger.info(f"Exporting ShareGPT format to directory:")
            logger.info(f"  - Shapes: {len(label_file.shapes)}")
            logger.info(f"  - Caption history: {len(caption_history)}")
            
            files_created = 0
            base_image_name = os.path.splitext(os.path.basename(label_file.imagePath or "image"))[0]
            
            # Create caption files (one per prompt-output pair)
            for i, entry in enumerate(caption_history):
                prompt = entry.get('prompt', '').strip()
                description = entry.get('description', '').strip()
                
                if prompt and description:
                    conversations = [
                        {
                            "from": "human",
                            "value": prompt
                        },
                        {
                            "from": "gpt", 
                            "value": description
                        }
                    ]
                    
                    # Create ShareGPT format data
                    sharegpt_data = {
                        "image": label_file.imagePath,
                        "task": "Caption",
                        "conversations": conversations
                    }
                    
                    # Write to file
                    filename = f"{base_image_name}_caption_{i+1:03d}.json"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
                    
                    files_created += 1
                    logger.info(f"Created caption file: {filename}")
            
            # Group shapes by task type for grounding files
            detection_shapes = []
            ocr_shapes = []
            
            for shape in label_file.shapes:
                # vlm_task is a direct property on shapes, not in other_data
                task_type = shape.get('vlm_task')
                
                if task_type == 'Detection':
                    detection_shapes.append(shape)
                elif task_type == 'OCR':
                    ocr_shapes.append(shape)
                # Only add shapes that have explicit task types
                # Don't default untagged shapes to any category
            
            # Create detection file if there are detection shapes
            if detection_shapes:
                unique_labels = list(set(shape['label'] for shape in detection_shapes))
                if len(unique_labels) == 1:
                    objects_text = unique_labels[0]
                else:
                    objects_text = ", ".join(unique_labels)
                
                human_prompt = f'Outline the position of each {objects_text} and output all the coordinates in JSON format {{"bbox_2d": [x1, y1, x2, y2], "label": "label"}}'
                gpt_response = self._format_shapes_as_json(detection_shapes, img_width, img_height)
                
                conversations = [
                    {
                        "from": "human",
                        "value": human_prompt
                    },
                    {
                        "from": "gpt",
                        "value": gpt_response
                    }
                ]
                
                # Create ShareGPT format data
                sharegpt_data = {
                    "image": label_file.imagePath,
                    "task": "Detection",
                    "conversations": conversations
                }
                
                # Write to file
                filename = f"{base_image_name}_detection.json"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
                
                files_created += 1
                logger.info(f"Created detection file: {filename}")
            
            # Create OCR file if there are OCR shapes
            if ocr_shapes:
                unique_labels = list(set(shape['label'] for shape in ocr_shapes))
                if len(unique_labels) == 1:
                    objects_text = unique_labels[0]
                else:
                    objects_text = ", ".join(unique_labels)
                
                human_prompt = f'Outline the position of each {objects_text} and output all the coordinates in JSON format {{"bbox_2d": [x1, y1, x2, y2], "label": "label"}}'
                gpt_response = self._format_shapes_as_json(ocr_shapes, img_width, img_height)
                
                conversations = [
                    {
                        "from": "human", 
                        "value": human_prompt
                    },
                    {
                        "from": "gpt",
                        "value": gpt_response
                    }
                ]
                
                # Create ShareGPT format data
                sharegpt_data = {
                    "image": label_file.imagePath,
                    "task": "OCR",
                    "conversations": conversations
                }
                
                # Write to file
                filename = f"{base_image_name}_ocr.json"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
                
                files_created += 1
                logger.info(f"Created OCR file: {filename}")
            
            logger.info(f"Successfully exported {files_created} ShareGPT files to: {output_dir}")
            return files_created
            
        except Exception as e:
            logger.error(f"Failed to export ShareGPT format to directory: {e}")
            return 0
    
    def export(self, label_file: LabelFile, output_filename: str) -> bool:
        """Export labelme data to ShareGPT format."""
        try:
            # Get image dimensions
            image_data = label_file.imageData
            if not image_data:
                logger.error(f"No image data available for export. LabelFile path: {label_file.imagePath}")
                return False
                
            import PIL.Image
            import io
            img_pil = PIL.Image.open(io.BytesIO(image_data))
            img_width, img_height = img_pil.size
            
            # Get current data
            prompt_history = label_file.otherData.get('prompt_history', [])
            caption_history = label_file.otherData.get('caption_history', [])
            
            logger.info(f"Generating ShareGPT format:")
            logger.info(f"  - Shapes: {len(label_file.shapes)}")
            logger.info(f"  - Caption history: {len(caption_history)}")
            
            conversations = []
            
            # Add caption conversations first (use actual prompts and outputs)
            if caption_history:
                for entry in caption_history:
                    prompt = entry.get('prompt', '').strip()
                    description = entry.get('description', '').strip()
                    if prompt and description:
                        conversations.extend([
                            {
                                "from": "human",
                                "value": prompt
                            },
                            {
                                "from": "gpt", 
                                "value": description
                            }
                        ])
            
            # Add detection/OCR conversations (generate prompts and JSON responses)
            if label_file.shapes:
                # Group shapes by task type
                detection_shapes = []
                ocr_shapes = []
                
                for shape in label_file.shapes:
                    task_type = None
                    if hasattr(shape, 'other_data') and shape.get('other_data'):
                        task_type = shape['other_data'].get('vlm_task')
                    
                    if task_type == 'Detection':
                        detection_shapes.append(shape)
                    elif task_type == 'OCR':
                        ocr_shapes.append(shape)
                    else:
                        # Default to detection for shapes without task type
                        detection_shapes.append(shape)
                
                # Add detection conversation
                if detection_shapes:
                    unique_labels = list(set(shape['label'] for shape in detection_shapes))
                    if len(unique_labels) == 1:
                        objects_text = unique_labels[0]
                    else:
                        objects_text = ", ".join(unique_labels)
                    
                    human_prompt = f'Outline the position of each {objects_text} and output all the coordinates in JSON format {{"bbox_2d": [x1, y1, x2, y2], "label": "label"}}'
                    gpt_response = self._format_shapes_as_json(detection_shapes, img_width, img_height)
                    
                    conversations.extend([
                        {
                            "from": "human",
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": gpt_response
                        }
                    ])
                
                # Add OCR conversation
                if ocr_shapes:
                    unique_labels = list(set(shape['label'] for shape in ocr_shapes))
                    if len(unique_labels) == 1:
                        objects_text = unique_labels[0]
                    else:
                        objects_text = ", ".join(unique_labels)
                    
                    human_prompt = f'Outline the position of each {objects_text} and output all the coordinates in JSON format {{"bbox_2d": [x1, y1, x2, y2], "label": "label"}}'
                    gpt_response = self._format_shapes_as_json(ocr_shapes, img_width, img_height)
                    
                    conversations.extend([
                        {
                            "from": "human", 
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": gpt_response
                        }
                    ])
            
            # Determine task type
            task_type = "Caption"
            if label_file.shapes:
                detection_count = sum(1 for shape in label_file.shapes if shape.get('other_data', {}).get('vlm_task') == 'Detection')
                ocr_count = sum(1 for shape in label_file.shapes if shape.get('other_data', {}).get('vlm_task') == 'OCR')
                if detection_count > 0:
                    task_type = "Detection"
                elif ocr_count > 0:
                    task_type = "OCR"
            
            # Create ShareGPT format data
            sharegpt_data = {
                "image": label_file.imagePath,
                "task": task_type,
                "conversations": conversations
            }
            
            # Write to file
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Exported ShareGPT format to: {output_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export ShareGPT format: {e}")
            return False
    
    def _format_shapes_as_json(self, shapes: List[Dict], img_width: int, img_height: int) -> str:
        """Format labelme shapes as JSON response for ShareGPT."""
        if not shapes:
            return ""
        
        json_objects = []
        
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            shape_type = shape['shape_type']
            
            # Convert to normalized coordinates [0-1]
            if shape_type == 'rectangle' and len(points) >= 2:
                x1_norm = points[0][0] / img_width
                y1_norm = points[0][1] / img_height  
                x2_norm = points[1][0] / img_width
                y2_norm = points[1][1] / img_height
                
                json_objects.append({
                    "bbox_2d": [round(x1_norm, 3), round(y1_norm, 3), round(x2_norm, 3), round(y2_norm, 3)],
                    "label": label
                })
            elif shape_type == 'polygon' and len(points) >= 3:
                # For polygons, compute bounding box
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width 
                y2_norm = y2 / img_height
                
                json_objects.append({
                    "bbox_2d": [round(x1_norm, 3), round(y1_norm, 3), round(x2_norm, 3), round(y2_norm, 3)],
                    "label": label
                })
        
        # Format as individual JSON objects, one per line
        json_lines = []
        for obj in json_objects:
            json_lines.append(json.dumps(obj, separators=(',', ': ')))
        
        return ','.join(json_lines)


def is_conversation_format(filename: str) -> bool:
    """Check if a file is in conversation format."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            # Must have conversations field with list
            if 'conversations' not in data:
                return False
                
            conversations = data['conversations']
            if not isinstance(conversations, list):
                return False
                
            # Check if conversations have proper structure
            if conversations:
                for conv in conversations:
                    if not isinstance(conv, dict):
                        return False
                    if 'from' not in conv or 'value' not in conv:
                        return False
                    if conv.get('from') not in ['human', 'gpt']:
                        return False
                        
            return True
            
        return False
        
    except Exception as e:
        logger.debug(f"Failed to check conversation format for {filename}: {e}")
        return False


def analyze_conversation_file(filename: str) -> Optional[Dict]:
    """Analyze a conversation file and return statistics about conversation types."""
    try:
        loader = ConversationFormatLoader()
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = data.get('conversations', [])
        stats = loader.analyze_conversations(conversations)
        grounding_convs, text_convs = loader.separate_conversation_types(conversations)
        
        return {
            'filename': filename,
            'image': data.get('image', 'N/A'),
            'total_conversations': stats['total_conversations'],
            'grounding_conversations': stats['grounding_conversations'],
            'pure_text_conversations': stats['pure_text_conversations'],
            'total_annotations': stats['total_annotations'],
            'grounding_conversation_pairs': len(grounding_convs) // 2,
            'text_conversation_pairs': len(text_convs) // 2,
            'has_spatial_annotations': stats['total_annotations'] > 0
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze conversation file {filename}: {e}")
        return None