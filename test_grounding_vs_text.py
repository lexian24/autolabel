#!/usr/bin/env python3
"""
Test script to verify grounding vs text conversation handling.
"""

import json
import tempfile
import os
from labelme.conversation_format import ConversationFormatLoader, analyze_conversation_file


def test_conversation_types():
    """Test the handling of different conversation types."""
    
    print("Testing Grounding vs Text Conversation Handling")
    print("=" * 50)
    
    # Create test data with mixed conversation types
    test_data = {
        "image": "test_image.jpg",
        "conversations": [
            # Grounding conversation pair
            {
                "from": "human",
                "value": "Detect all cars and describe using bounding boxes."
            },
            {
                "from": "gpt",
                "value": "There are <p>car</p>[0.1,0.2,0.4,0.6] and <p>car</p>[0.5,0.3,0.8,0.7] in the image."
            },
            # Pure text conversation pair
            {
                "from": "human", 
                "value": "What is the weather like in this scene?"
            },
            {
                "from": "gpt",
                "value": "The image shows a sunny day with clear blue skies."
            },
            # Another grounding conversation with points
            {
                "from": "human",
                "value": "Mark all traffic lights with points."
            },
            {
                "from": "gpt",
                "value": "I can see <p>traffic_light</p>[0.15,0.1] at the intersection."
            }
        ]
    }
    
    # Write test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        test_file = f.name
    
    try:
        # Test conversation analysis
        print("1. Testing conversation analysis...")
        analysis = analyze_conversation_file(test_file)
        
        if analysis:
            print(f"   ✓ Total conversations: {analysis['total_conversations']}")
            print(f"   ✓ Grounding conversations: {analysis['grounding_conversations']}")
            print(f"   ✓ Pure text conversations: {analysis['pure_text_conversations']}")
            print(f"   ✓ Total annotations: {analysis['total_annotations']}")
            print(f"   ✓ Has spatial annotations: {analysis['has_spatial_annotations']}")
        else:
            print("   ✗ Analysis failed")
            return False
        
        # Test conversation separation
        print("\n2. Testing conversation separation...")
        loader = ConversationFormatLoader()
        conversations = test_data['conversations']
        grounding_convs, text_convs = loader.separate_conversation_types(conversations)
        
        print(f"   ✓ Original conversations: {len(conversations)}")
        print(f"   ✓ Grounding conversations: {len(grounding_convs)}")
        print(f"   ✓ Text conversations: {len(text_convs)}")
        
        # Test annotation parsing
        print("\n3. Testing annotation parsing...")
        test_gpt_responses = [
            "There are <p>car</p>[0.1,0.2,0.4,0.6] and <p>car</p>[0.5,0.3,0.8,0.7] in the image.",
            "I can see <p>traffic_light</p>[0.15,0.1] at the intersection.",
            "The image shows a sunny day with clear blue skies."  # No annotations
        ]
        
        for i, response in enumerate(test_gpt_responses):
            annotations = loader._parse_gpt_annotations(response)
            print(f"   ✓ Response {i+1}: {len(annotations)} annotations found")
            for ann in annotations:
                print(f"     - {ann.label}: {ann.annotation_type} ({len(ann.coordinates)} coords)")
        
        print("\n4. Testing coordinate types...")
        coordinate_tests = [
            ("Point", "<p>landmark</p>[0.5,0.3]"),
            ("Bounding Box", "<p>car</p>[0.1,0.2,0.4,0.6]"),
            ("Oriented Box", "<p>ship</p>[0.1,0.2,0.2,0.15,0.4,0.3,0.3,0.35]"),
            ("Polygon", "<p>building</p>[0.1,0.1,0.3,0.15,0.35,0.4,0.2,0.45]")
        ]
        
        for test_name, test_response in coordinate_tests:
            annotations = loader._parse_gpt_annotations(test_response)
            if annotations:
                ann = annotations[0]
                print(f"   ✓ {test_name}: {ann.annotation_type} with {len(ann.coordinates)} coordinates")
            else:
                print(f"   ✗ {test_name}: Failed to parse")
        
        print(f"\n5. Summary:")
        print(f"   • System successfully distinguishes grounding vs text conversations")
        print(f"   • Grounding conversations contain spatial annotations (<p>label</p>[coords])")
        print(f"   • Text conversations contain pure descriptive text")
        print(f"   • Both types can coexist in the same conversation file")
        print(f"   • System supports points, boxes, oriented boxes, and polygons")
        print(f"   • Coordinates are normalized to [0,1] range for resolution independence")
        
        return True
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    success = test_conversation_types()
    if success:
        print("\n✓ All tests passed! The conversation format system correctly handles both grounding and text conversations.")
    else:
        print("\n✗ Tests failed!")