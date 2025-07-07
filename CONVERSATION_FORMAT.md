# Conversation Format Documentation

## Overview

The conversation format is a JSON-based annotation format commonly used for training Vision-Language Models (VLMs). It supports structured conversations between humans and AI assistants about images, including both **grounding conversations** (with spatial annotations) and **pure text conversations** (descriptive text only).

## Format Specification

### Basic Structure

```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "Prompt or question from human"
        },
        {
            "from": "gpt", 
            "value": "Response from AI assistant"
        }
    ]
}
```

## Conversation Types

### 1. Grounding Conversations

**Grounding conversations** contain spatial annotations with coordinates embedded in the GPT responses. These are used for training models that need to understand spatial relationships and object locations.

**Format**: `<p>label</p>[coordinates]`

**Examples**:

```json
{
    "from": "human",
    "value": "Detect all cars in the image and describe using bounding boxes."
},
{
    "from": "gpt",
    "value": "There are <p>car</p>[0.1,0.2,0.4,0.6] and <p>car</p>[0.5,0.3,0.8,0.7] in the image."
}
```

**Supported Annotation Types**:
- **Points**: `<p>landmark</p>[x,y]`
- **Bounding Boxes**: `<p>object</p>[x1,y1,x2,y2]`
- **Oriented Boxes**: `<p>ship</p>[x1,y1,x2,y2,x3,y3,x4,y4]`
- **Polygons**: `<p>building</p>[x1,y1,x2,y2,...,xn,yn]`

**Use Cases**:
- Object detection training
- Instance segmentation
- Keypoint detection
- Spatial reasoning tasks

### 2. Pure Text Conversations

**Pure text conversations** contain only descriptive text without spatial coordinates. These are used for training general vision-language understanding.

**Examples**:

```json
{
    "from": "human",
    "value": "Describe what you see in the image."
},
{
    "from": "gpt",
    "value": "I see a beautiful landscape with mountains in the background, a lake in the foreground, and trees around the shoreline. The lighting suggests early morning."
}
```

**Use Cases**:
- Image captioning
- Visual question answering
- Scene understanding
- Attribute recognition

### 3. Mixed Conversations

A single conversation file can contain both types:

```json
{
    "image": "street_scene.jpg",
    "conversations": [
        // Grounding conversation
        {
            "from": "human",
            "value": "Detect all cars in the image."
        },
        {
            "from": "gpt", 
            "value": "There are <p>car</p>[0.1,0.2,0.4,0.6] and <p>car</p>[0.5,0.3,0.8,0.7] in the image."
        },
        // Pure text conversation
        {
            "from": "human",
            "value": "What is the weather like?"
        },
        {
            "from": "gpt",
            "value": "The image shows a sunny day with clear skies and good visibility."
        }
    ]
}
```

## System Capabilities

### Automatic Analysis and Separation

The system can automatically:

1. **Analyze conversation types**:
   ```python
   from labelme.conversation_format import analyze_conversation_file
   
   analysis = analyze_conversation_file("conversation.json")
   print(f"Grounding conversations: {analysis['grounding_conversations']}")
   print(f"Text conversations: {analysis['pure_text_conversations']}")
   ```

2. **Separate conversation types**:
   ```python
   loader = ConversationFormatLoader()
   grounding_convs, text_convs = loader.separate_conversation_types(conversations)
   ```

3. **Parse spatial annotations**:
   ```python
   annotations = loader._parse_gpt_annotations(gpt_response)
   for ann in annotations:
       print(f"{ann.label}: {ann.annotation_type} at {ann.coordinates}")
   ```

### Coordinate System

All coordinates are **normalized** to the range [0, 1]:
- `0.0` = left/top edge of image
- `1.0` = right/bottom edge of image

This ensures resolution independence and consistent behavior across different image sizes.

**Conversion**:
- **Loading**: Normalized coordinates × image dimensions = pixel coordinates
- **Exporting**: Pixel coordinates ÷ image dimensions = normalized coordinates

## Usage in Labelme

### Loading Conversation Files

1. **Automatic Detection**: Labelme automatically detects conversation format files
2. **Annotation Extraction**: Spatial annotations are converted to labelme shapes
3. **Conversation Preservation**: Original conversations are stored in `otherData`
4. **Analysis Logging**: System logs conversation type statistics

```python
# When loading a conversation file, labelme:
# 1. Detects format automatically
# 2. Separates grounding from text conversations  
# 3. Converts spatial annotations to shapes
# 4. Preserves all conversation data
```

### Exporting to Conversation Format

1. **Shape Conversion**: Labelme shapes → normalized coordinates
2. **Conversation Generation**: Automatic prompt/response generation
3. **Type Preservation**: Maintains existing text conversations if present
4. **Flexible Export**: Option to include/exclude text conversations

```python
# Export options:
exporter.export(label_file, "output.json", include_text_conversations=True)
```

## Common Usage Patterns

### Object Detection Training
```json
{
    "from": "human",
    "value": "Detect all [objects] shown in the image and describe using bounding boxes."
},
{
    "from": "gpt",
    "value": "There is <p>object1</p>[x1,y1,x2,y2] and <p>object2</p>[x1,y1,x2,y2] in the image."
}
```

### Scene Description Training
```json
{
    "from": "human", 
    "value": "Describe the scene in detail."
},
{
    "from": "gpt",
    "value": "This is an urban scene with multiple vehicles on a busy street..."
}
```

### Multi-modal Training
```json
{
    "conversations": [
        // First: Detection task (grounding)
        {"from": "human", "value": "Find all people in the image."},
        {"from": "gpt", "value": "I can see <p>person</p>[0.2,0.3,0.4,0.8] in the image."},
        
        // Second: Description task (text)
        {"from": "human", "value": "What are they doing?"},
        {"from": "gpt", "value": "The person appears to be walking down the street."}
    ]
}
```

## Best Practices

### For Grounding Conversations:
- Use clear, specific prompts mentioning annotation type
- Ensure coordinates are properly normalized
- Include sufficient context in responses
- Use consistent label naming

### For Text Conversations:
- Ask open-ended descriptive questions
- Encourage detailed, informative responses
- Focus on visual content and relationships
- Avoid spatial coordinate references

### For Mixed Datasets:
- Balance grounding and text conversations
- Ensure logical conversation flow
- Maintain consistency in terminology
- Consider task-specific conversation patterns

## Troubleshooting

### Common Issues:

1. **Coordinates out of range**: Ensure all coordinates are in [0, 1] range
2. **Malformed annotations**: Check `<p>label</p>[coordinates]` format
3. **Missing images**: Verify image paths are correct (relative or absolute)
4. **Parse errors**: Validate JSON format and conversation structure

### Analysis Tools:

```python
# Analyze conversation file
analysis = analyze_conversation_file("file.json")
if analysis:
    print(f"Total annotations: {analysis['total_annotations']}")
    print(f"Has grounding: {analysis['has_spatial_annotations']}")
```

## Implementation Details

The conversation format support is implemented in `labelme/conversation_format.py` with:

- `ConversationFormatLoader`: Handles loading and parsing
- `ConversationFormatExporter`: Handles export and conversion  
- `analyze_conversation_file()`: Provides conversation analysis
- `separate_conversation_types()`: Separates grounding from text conversations

The system integrates with labelme's main application through:
- Automatic format detection in file loading
- Export menu option for conversation format
- Preservation of conversation data in label files