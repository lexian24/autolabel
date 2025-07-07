#!/usr/bin/env python3
"""
Directory ShareGPT Converter

Converts a directory of labelme JSON files to ShareGPT format.
This script processes all .json files in the input directory and converts them
to ShareGPT format using the same logic as the labelme export function.

Usage:
    python convert_directory_to_sharegpt.py <input_dir> <output_dir>

Requirements:
    - PIL (pillow)
    - labelme package with ShareGPTExporter
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_image_for_json(json_path):
    """Find the corresponding image file for a JSON annotation file."""
    json_path = Path(json_path)
    json_stem = json_path.stem
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Look in the same directory
    for ext in image_extensions:
        image_path = json_path.parent / (json_stem + ext)
        if image_path.exists():
            return str(image_path)
    
    # Look for common image directory structures
    possible_dirs = [
        json_path.parent / 'images',
        json_path.parent / 'JPEGImages', 
        json_path.parent / 'raw',
        json_path.parent.parent / 'images',
        json_path.parent.parent / 'JPEGImages'
    ]
    
    for img_dir in possible_dirs:
        if img_dir.exists() and img_dir.is_dir():
            for ext in image_extensions:
                image_path = img_dir / (json_stem + ext)
                if image_path.exists():
                    return str(image_path)
    
    return None

def create_labelfile_from_json(json_path, image_path=None):
    """Create a LabelFile object from a JSON annotation file."""
    try:
        from labelme.label_file import LabelFile
        import PIL.Image
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create LabelFile object
        label_file = LabelFile()
        
        # Set basic properties
        label_file.imagePath = image_path or json_path.replace('.json', '.jpg')
        label_file.shapes = data.get('shapes', [])
        label_file.otherData = data.get('otherData', {})
        
        # Handle caption_history if it exists
        if 'caption_history' in data:
            label_file.otherData['caption_history'] = data['caption_history']
        
        # Load image data if image exists
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                label_file.imageData = f.read()
            
            # Verify image can be opened
            try:
                import io
                img_pil = PIL.Image.open(io.BytesIO(label_file.imageData))
                logger.debug(f"Loaded image {image_path}: {img_pil.size}")
            except Exception as e:
                logger.warning(f"Failed to verify image {image_path}: {e}")
                
        else:
            logger.warning(f"No image found for {json_path}")
            # Create a dummy 1x1 image for export
            import io
            from PIL import Image
            dummy_img = Image.new('RGB', (1, 1), color='white')
            img_bytes = io.BytesIO()
            dummy_img.save(img_bytes, format='PNG')
            label_file.imageData = img_bytes.getvalue()
        
        return label_file
        
    except Exception as e:
        logger.error(f"Failed to create LabelFile from {json_path}: {e}")
        return None

def convert_file_to_sharegpt(json_path, output_dir):
    """Convert a single JSON file to ShareGPT format."""
    try:
        # Find corresponding image
        image_path = find_image_for_json(json_path)
        if not image_path:
            logger.warning(f"No image found for {json_path}")
        
        # Create LabelFile object
        label_file = create_labelfile_from_json(json_path, image_path)
        if not label_file:
            return False
        
        # Generate output filename
        json_name = Path(json_path).stem
        output_path = Path(output_dir) / f"{json_name}_sharegpt.json"
        
        # Export using ShareGPTExporter
        from labelme.conversation_format import ShareGPTExporter
        exporter = ShareGPTExporter()
        
        success = exporter.export(label_file, str(output_path))
        
        if success:
            logger.info(f"Converted: {json_path} -> {output_path}")
            return True
        else:
            logger.error(f"Failed to export: {json_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error converting {json_path}: {e}")
        return False

def convert_directory(input_dir, output_dir):
    """Convert all JSON files in input_dir to ShareGPT format in output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return False
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return False
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_path.rglob('*.json'))
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return False
    
    logger.info(f"Found {len(json_files)} JSON files to convert")
    
    # Convert each file
    success_count = 0
    for json_file in json_files:
        try:
            # Skip if it's already a ShareGPT file
            if '_sharegpt' in json_file.stem:
                logger.debug(f"Skipping ShareGPT file: {json_file}")
                continue
                
            # Check if it's a labelme format file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip if it's already in ShareGPT format
            if 'conversations' in data and 'task' in data:
                logger.debug(f"Skipping ShareGPT format file: {json_file}")
                continue
                
            # Skip if it doesn't look like labelme format
            if 'shapes' not in data and 'caption_history' not in data:
                logger.debug(f"Skipping non-labelme file: {json_file}")
                continue
            
            if convert_file_to_sharegpt(str(json_file), str(output_path)):
                success_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    
    logger.info(f"Successfully converted {success_count}/{len(json_files)} files")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(
        description="Convert directory of labelme JSON files to ShareGPT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all JSON files in data/ to ShareGPT format in output/
    python convert_directory_to_sharegpt.py data/ output/
    
    # Convert with verbose logging
    python convert_directory_to_sharegpt.py -v data/ output/
        """)
    
    parser.add_argument('input_dir', help='Input directory containing labelme JSON files')
    parser.add_argument('output_dir', help='Output directory for ShareGPT format files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Converting labelme files from {args.input_dir} to ShareGPT format in {args.output_dir}")
    
    try:
        success = convert_directory(args.input_dir, args.output_dir)
        if success:
            logger.info("Directory conversion completed successfully!")
            sys.exit(0)
        else:
            logger.error("Directory conversion failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 