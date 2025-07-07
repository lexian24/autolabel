#!/usr/bin/env python3
"""
Example script demonstrating conversation format support in labelme.

This script shows how the system handles two types of conversations:
1. Grounding conversations: Include spatial annotations like <p>label</p>[coordinates]
2. Pure text conversations: Descriptive text without spatial coordinates

The system can analyze, separate, and handle both types appropriately.
"""

import json
import os
from labelme.conversation_format import (
    ConversationFormatLoader, 
    ConversationFormatExporter,
    analyze_conversation_file,
    is_conversation_format
)


def create_mixed_conversation_example():
    """Create an example conversation file with both grounding and text conversations."""
    
    # Example with mixed conversation types
    conversation_data = {
        "image": "example_image.jpg",
        "conversations": [
            # Grounding conversation pair (human + gpt with coordinates)
            {
                "from": "human",
                "value": "Detect all cars in the image and describe using bounding boxes."
            },
            {
                "from": "gpt", 
                "value": "There are <p>car</p>[0.1,0.2,0.3,0.4] and <p>car</p>[0.5,0.6,0.7,0.8] in the image."
            },
            
            # Pure text conversation pair (human + gpt without coordinates)
            {
                "from": "human",
                "value": "What is the weather like in this image?"
            },
            {
                "from": "gpt",
                "value": "The image shows a sunny day with clear blue skies and good visibility."
            },
            
            # Another grounding conversation with different annotation type
            {
                "from": "human", 
                "value": "Mark all the traffic lights with points."
            },
            {
                "from": "gpt",
                "value": "I can see <p>traffic_light</p>[0.15,0.1] at the intersection."
            },
            
            # Another pure text conversation
            {
                "from": "human",
                "value": "Describe the overall scene composition."
            },
            {
                "from": "gpt",
                "value": "This is an urban street scene with multiple vehicles, traffic infrastructure, and buildings in the background. The perspective appears to be from a driver's viewpoint."
            }
        ]
    }
    
    return conversation_data


def create_grounding_only_example():
    """Create an example with only grounding conversations."""
    return {
        "image": "detection_image.jpg", 
        "conversations": [
            {
                "from": "human",
                "value": "Detect all objects in the image and describe using oriented bounding boxes."
            },
            {
                "from": "gpt",
                "value": "There is <p>ship</p>[0.38,0.12,0.42,0.04,0.85,0.27,0.80,0.35] and <p>boat</p>[0.1,0.2,0.15,0.18,0.25,0.3,0.2,0.32] in the image."
            }
        ]
    }


def create_text_only_example():
    """Create an example with only text conversations."""
    return {
        "image": "description_image.jpg",
        "conversations": [
            {
                "from": "human", 
                "value": "Describe what you see in the image."
            },
            {
                "from": "gpt",
                "value": "I see a beautiful landscape with mountains in the background, a lake in the foreground, and some trees around the shoreline. The lighting suggests it's either early morning or late afternoon."
            },
            {
                "from": "human",
                "value": "What colors are dominant in this scene?"
            },
            {
                "from": "gpt", 
                "value": "The dominant colors are deep blues from the water and sky, various shades of green from the vegetation, and warm earth tones from the mountains and shoreline."
            }
        ]
    }


def demonstrate_conversation_analysis():
    """Demonstrate how the system analyzes and separates conversation types."""
    
    print("=== Conversation Format Analysis Demo ===\n")
    
    # Create examples
    examples = {
        "mixed_conversations.json": create_mixed_conversation_example(),
        "grounding_only.json": create_grounding_only_example(), 
        "text_only.json": create_text_only_example()
    }
    
    # Save examples to files
    for filename, data in examples.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Created example file: {filename}")
    
    print("\n" + "="*60 + "\n")
    
    # Analyze each example
    for filename in examples.keys():
        print(f"Analyzing: {filename}")
        print("-" * 40)
        
        # Check if it's conversation format
        is_conv_format = is_conversation_format(filename)
        print(f"Is conversation format: {is_conv_format}")
        
        if is_conv_format:
            # Get detailed analysis
            analysis = analyze_conversation_file(filename)
            if analysis:
                print(f"Image: {analysis['image']}")
                print(f"Total conversations: {analysis['total_conversations']}")
                print(f"Grounding conversations: {analysis['grounding_conversations']}")
                print(f"Pure text conversations: {analysis['pure_text_conversations']}") 
                print(f"Total annotations: {analysis['total_annotations']}")
                print(f"Grounding conversation pairs: {analysis['grounding_conversation_pairs']}")
                print(f"Text conversation pairs: {analysis['text_conversation_pairs']}")
                print(f"Has spatial annotations: {analysis['has_spatial_annotations']}")
                
                # Demonstrate conversation separation
                loader = ConversationFormatLoader()
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                conversations = data.get('conversations', [])
                grounding_convs, text_convs = loader.separate_conversation_types(conversations)
                
                print(f"\nSeparated conversations:")
                print(f"  - Grounding conversations: {len(grounding_convs)} entries")
                print(f"  - Text conversations: {len(text_convs)} entries")
                
                if grounding_convs:
                    print(f"\nFirst grounding conversation:")
                    for i, conv in enumerate(grounding_convs[:2]):  # Show first pair
                        print(f"    {conv['from']}: {conv['value'][:60]}...")
                        
                if text_convs:
                    print(f"\nFirst text conversation:")
                    for i, conv in enumerate(text_convs[:2]):  # Show first pair
                        print(f"    {conv['from']}: {conv['value'][:60]}...")
        
        print("\n" + "="*60 + "\n")
    
    # Cleanup
    for filename in examples.keys():
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up: {filename}")


def demonstrate_annotation_types():
    """Demonstrate different annotation types supported in grounding conversations."""
    
    print("=== Annotation Types Demo ===\n")
    
    annotation_examples = {
        "points": {
            "description": "Point annotations [x,y]",
            "example": "<p>landmark</p>[0.5,0.3]"
        },
        "bounding_boxes": {
            "description": "Rectangular bounding boxes [x1,y1,x2,y2]", 
            "example": "<p>car</p>[0.1,0.2,0.4,0.6]"
        },
        "oriented_boxes": {
            "description": "Oriented bounding boxes [x1,y1,x2,y2,x3,y3,x4,y4]",
            "example": "<p>ship</p>[0.1,0.2,0.2,0.15,0.4,0.3,0.3,0.35]"
        },
        "polygons": {
            "description": "Polygon annotations [x1,y1,x2,y2,...,xn,yn]",
            "example": "<p>building</p>[0.1,0.1,0.3,0.15,0.35,0.4,0.2,0.45,0.05,0.3]"
        }
    }
    
    for ann_type, info in annotation_examples.items():
        print(f"{ann_type.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Format: {info['example']}")
        print()
    
    # Test parsing of different annotation types
    loader = ConversationFormatLoader()
    
    print("Parsing test:")
    test_gpt_response = (
        "I can see multiple objects: "
        "<p>car</p>[0.1,0.2,0.4,0.6] and "
        "<p>person</p>[0.7,0.8] and "
        "<p>building</p>[0.0,0.0,0.2,0.1,0.25,0.4,0.05,0.45]"
    )
    
    annotations = loader._parse_gpt_annotations(test_gpt_response)
    print(f"Parsed {len(annotations)} annotations:")
    for ann in annotations:
        print(f"  - {ann.label}: {ann.annotation_type} with {len(ann.coordinates)} coordinates")


def demonstrate_usage_patterns():
    """Demonstrate common usage patterns for grounding vs text conversations."""
    
    print("=== Usage Patterns ===\n")
    
    patterns = {
        "Object Detection": {
            "type": "grounding",
            "human_prompt": "Detect all [objects] in the image.",
            "gpt_response": "Contains <p>label</p>[coordinates] annotations",
            "use_case": "Training object detection models"
        },
        
        "Instance Segmentation": {
            "type": "grounding", 
            "human_prompt": "Segment all [objects] using polygons.",
            "gpt_response": "Contains <p>label</p>[polygon_coordinates] annotations",
            "use_case": "Training segmentation models"
        },
        
        "Keypoint Detection": {
            "type": "grounding",
            "human_prompt": "Mark key points on [objects].",
            "gpt_response": "Contains <p>keypoint_name</p>[x,y] annotations", 
            "use_case": "Training pose estimation models"
        },
        
        "Scene Description": {
            "type": "text",
            "human_prompt": "Describe what you see in the image.",
            "gpt_response": "Pure text description without coordinates",
            "use_case": "Training vision-language models for captioning"
        },
        
        "Visual Question Answering": {
            "type": "text",
            "human_prompt": "What is [specific question] in the image?",
            "gpt_response": "Text answer based on visual content",
            "use_case": "Training VQA models"
        },
        
        "Attribute Recognition": {
            "type": "text",
            "human_prompt": "What are the attributes of objects in this image?",
            "gpt_response": "Text describing colors, sizes, states, etc.",
            "use_case": "Training attribute recognition models"
        }
    }
    
    for pattern_name, info in patterns.items():
        print(f"{pattern_name.upper()} ({info['type']} conversation):")
        print(f"  Human: {info['human_prompt']}")
        print(f"  GPT: {info['gpt_response']}")
        print(f"  Use case: {info['use_case']}")
        print()


if __name__ == "__main__":
    print("Conversation Format Demonstration")
    print("="*50)
    print()
    
    demonstrate_conversation_analysis()
    demonstrate_annotation_types() 
    demonstrate_usage_patterns()
    
    print("\nKey Points:")
    print("- Grounding conversations contain spatial annotations with coordinates")
    print("- Text conversations contain pure descriptive text without coordinates") 
    print("- The system can automatically separate and analyze both types")
    print("- Both types can coexist in the same conversation file")
    print("- Coordinates are normalized (0-1 range) for resolution independence")
    print("- The system supports points, bounding boxes, oriented boxes, and polygons")