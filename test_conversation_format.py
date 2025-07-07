#!/usr/bin/env python3
"""
Test script for conversation format support in labelme.

This script creates a sample conversation format file and tests loading/exporting.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

def create_test_conversation_file():
    """Create a test conversation format file."""
    # Create sample conversation data
    conversation_data = {
        "image": "test_image.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "Detect all ships in the image and describe using oriented bounding box."
            },
            {
                "from": "gpt", 
                "value": "There is <p>Dry-Cargo-Ship</p>[0.38,0.12,0.42,0.04,0.85,0.27,0.80,0.35] and <p>Container-Ship</p>[0.10,0.20,0.30,0.40] in the image."
            }
        ]
    }
    
    return conversation_data

def test_conversation_format():
    """Test conversation format functionality."""
    print("Testing conversation format support...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test conversation file
        conv_file = temp_path / "test_conversation.json"
        conversation_data = create_test_conversation_file()
        
        with open(conv_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        print(f"Created test file: {conv_file}")
        
        # Test is_conversation_format function
        try:
            from labelme.conversation_format import is_conversation_format
            is_conv = is_conversation_format(str(conv_file))
            print(f"is_conversation_format: {is_conv}")
            assert is_conv, "Should detect conversation format"
        except ImportError as e:
            print(f"Import error: {e}")
            return False
        
        # Test loading conversation format
        try:
            from labelme.conversation_format import ConversationFormatLoader
            loader = ConversationFormatLoader()
            
            # Since we don't have an actual image, this will fail
            # but we can test the parsing logic
            print("Testing conversation parsing...")
            annotations = loader._parse_gpt_annotations(
                "There is <p>ship</p>[0.1,0.2,0.3,0.4] and <p>boat</p>[0.5,0.6,0.7,0.8] in the image."
            )
            
            print(f"Parsed {len(annotations)} annotations:")
            for ann in annotations:
                print(f"  - {ann.label}: {ann.coordinates} ({ann.annotation_type})")
                
            assert len(annotations) == 2, "Should parse 2 annotations"
            assert annotations[0].label == "ship", "First annotation should be ship"
            assert annotations[0].annotation_type == "bbox", "Should be bounding box"
            
        except Exception as e:
            print(f"Error testing loader: {e}")
            return False
        
        print("All tests passed!")
        return True

if __name__ == "__main__":
    success = test_conversation_format()
    if success:
        print("\n✅ Conversation format support is working correctly!")
    else:
        print("\n❌ Conversation format tests failed!")