"""
Core Qwen2.5-VL model handler for vision-language tasks.
"""

import time
from typing import List, Dict, Optional, Union
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from loguru import logger


class QwenVLModel:
    """
    Qwen2.5-VL model wrapper for vision-language inference.
    
    Handles model loading, device management, and inference operations.
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Initialize the Qwen VL model.
        
        Args:
            model_path: HuggingFace model path or local model directory
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self._is_loaded = False
        
    def load_model(self) -> None:
        """Load the model and processor with optimal device/dtype settings."""
        if self._is_loaded:
            logger.debug("Model already loaded, skipping initialization")
            return
            
        logger.info(f"Loading Qwen2.5-VL model from {self.model_path}")
        
        # Select optimal device and dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.debug(f"Using device: {self.device}, dtype: {self.dtype}")
        
        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=self.dtype
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        self._is_loaded = True
        logger.info(f"Model loaded successfully on {self.device}")
        
    def inference(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 4096
    ) -> str:
        """
        Perform VLM inference on image and text prompt.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            prompt: Text prompt for the model
            system_prompt: System message for the model
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response from the model
        """
        if not self._is_loaded:
            self.load_model()
            
        # Convert image to PIL format if needed
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        # Prepare chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}, 
                {"image": pil_image}
            ]}
        ]
        
        # Apply chat template and tokenize
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text], 
            images=[pil_image], 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        # Decode generated tokens (excluding input tokens)
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Inference completed in {elapsed_time:.3f}s")
        
        return output_text
        
    def get_model_input_size(self, inputs: Dict) -> tuple[int, int]:
        """
        Get the model's input image dimensions for coordinate scaling.
        
        Args:
            inputs: Processed inputs from the processor
            
        Returns:
            (height, width) of model input image
        """
        if 'image_grid_thw' in inputs:
            grid_info = inputs['image_grid_thw'][0]
            height = grid_info[1] * 14  # Grid size factor
            width = grid_info[2] * 14
            return height, width
        else:
            # Fallback to default model input size
            return 448, 448  # Default Qwen2.5-VL input size
            
    def __del__(self):
        """Cleanup model resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global model instance (singleton pattern)
_global_model_instance: Optional[QwenVLModel] = None


def get_model_instance() -> QwenVLModel:
    """
    Get the global model instance (singleton pattern).
    
    Returns:
        Shared QwenVLModel instance
    """
    global _global_model_instance
    if _global_model_instance is None:
        _global_model_instance = QwenVLModel()
        _global_model_instance.load_model()
    return _global_model_instance 