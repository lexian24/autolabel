# labelme_VLM Refactoring Summary

## Overview

This document summarizes the labelme_VLM project to improve naming conventions, code organization, clarity, and maintainability.

## 📁 File Structure

```
labelme/
├── vlm/                          # 🆕 Dedicated VLM module
│   ├── __init__.py              # Clean exports
│   ├── qwen_model.py            # Core model management
│   ├── bbox_detection.py        # Object detection functionality
│   ├── description.py           # Image description functionality
│   └── utils.py                 # Common utilities and parsing
├── widgets/
│   ├── vlm_bbox_detection_widget.py  # 🆕 Renamed from AiLabelWidget
│   ├── vlm_prompt_widget.py          # 🆕 Renamed from AiPromptWidget
│   ├── ai_label_widget.py            # 📦 Legacy (kept for compatibility)
│   └── ai_prompt_widget.py           # 📦 Legacy (kept for compatibility)
└── _automation/
    └── bbox_from_text.py         # 📦 Legacy (kept for compatibility)
```

## 🔄 Function Name Changes

### Main App Functions
| Old Name | New Name | Purpose |
|----------|----------|---------|
| `submit_ai_label()` | `run_object_detection()` | VLM object detection |
| `on_add_prompt()` | `add_custom_vlm_prompt()` | Add custom VLM prompt |
| `on_edit_prompt()` | `edit_vlm_prompt()` | Edit existing prompt |
| `on_describe_shape()` | `describe_selected_bbox()` | Describe bbox contents |

### VLM Module Functions
| Old Location | New Location | New Name |
|--------------|--------------|----------|
| `bbox_from_text.get_vlm_shapes()` | `vlm.bbox_detection.detect_objects_with_vlm()` | Object detection |
| `bbox_from_text.inference()` | `vlm.description.get_image_description()` | Image description |
| `bbox_from_text.parse_qwen_output()` | `vlm.utils.parse_vlm_json_response()` | JSON parsing |

### Widget Classes
| Old Name | New Name | Purpose |
|----------|----------|---------|
| `AiLabelWidget` | `VlmBboxDetectionWidget` | Object detection UI |
| `AiPromptWidget` | `VlmPromptWidget` | Custom prompting UI |

### UI Actions
| Old Name | New Name | Purpose |
|----------|----------|---------|
| `describeAction` | `describeBboxAction` | Bbox description action |

## 🏗️ Architectural Improvements

### 1. **Centralized Model Management**
```python
# Before: Model loaded multiple times
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)

# After: Singleton pattern
model = get_model_instance()  # Reuses existing instance
```

### 2. **Unified Response Parsing**
```python
# Before: Different parsing logic scattered
def parse_qwen_output(text): # Multiple implementations

# After: Single comprehensive parser
def parse_vlm_json_response(text) -> Tuple[List[Dict], str]:
    # Handles multiple JSON schemas consistently
```

### 3. **Clear Separation of Concerns**
- **`qwen_model.py`**: Pure model operations
- **`bbox_detection.py`**: Object detection logic
- **`description.py`**: Text description logic
- **`utils.py`**: Common utilities

### 4. **Improved Error Handling**
```python
# Before: Basic error messages
except Exception as e:
    self.errorMessage("Error", str(e))

# After: Descriptive, user-friendly errors
except Exception as e:
    logger.error(f"Object detection failed: {e}")
    self.errorMessage(
        self.tr("Detection Failed"), 
        self.tr(f"Object detection failed:\n{e}")
    )
```

## 🎨 UI Improvements

### 1. **Better Widget Names**
- `VlmBboxDetectionWidget`: Clear purpose (object detection)
- `VlmPromptWidget`: Clear purpose (custom prompting)

### 2. **Enhanced Tooltips and Labels**
```python
# Before: "AI label"
# After: "Objects to Detect" with helpful tooltip
```

### 3. **Improved Action Names**
```python
# Before: "Describe"
# After: "Describe Contents" with tooltip "Use VLM to describe contents of the selected bounding box"
```

## 📚 Key Features Maintained

### 1. **Object Detection (AI Labeling)**
- Enter object names → Get automatic bounding boxes
- Uses toolbar widget for input
- Creates labelme Shape objects

### 2. **Custom VLM Prompting**
- Ask custom questions about images
- Maintains prompt/description history
- Editable and removable prompts

### 3. **Bbox Description**
- Right-click any rectangle → Get VLM description
- Highlights bbox in red for context
- Adds to prompt history with bbox coordinates


### 1. **Type Hints & Documentation**
```python
def detect_objects_with_vlm(
    image: np.ndarray, 
    objects: str
) -> Tuple[List[Dict], str]:
    """
    Detect objects in an image and return bounding boxes.
    
    Args:
        image: Input image as numpy array (H, W, C)
        objects: Comma-separated object names
        
    Returns:
        (shapes_list, description_text)
    """
```

### 2. **Consistent Logging**
```python
logger.info(f"Detected {len(shapes)} objects")
logger.error(f"Object detection failed: {e}")
```

### 3. **Better Resource Management**
```python
def __del__(self):
    """Cleanup model resources."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```


2. **Widget Usage**:
   ```python
   # Old
   widget = AiLabelWidget(on_submit=callback)
   
   # New
   widget = VlmBboxDetectionWidget(on_detect_callback=callback)
   ```
