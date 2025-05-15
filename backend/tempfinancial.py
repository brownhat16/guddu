# enhanced_financial_image_analyzer.py

import base64
import json
import os
import requests
import time
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import re


class ObjectDetectionModule:
    def __init__(self,
                 yolo_model_path: str = "yolov9-e.pt",
                 detr_model_name: str = "facebook/detr-resnet-50",
                 use_deyo: bool = False):
        """Initialize object detection models

        Args:
            yolo_model_path: Path to YOLO model weights
            detr_model_name: HuggingFace model name for DETR
            use_deyo: Whether to use DEYO architecture
        """
        self.yolo_model_path = yolo_model_path
        self.detr_model_name = detr_model_name
        self.use_deyo = use_deyo

        # Initialize models lazily to avoid loading them if not used
        self.yolo_model = None
        self.detr_model = None
        self.detr_processor = None
        self.deyo_model = None

        print(f"Object Detection Module initialized with:")
        print(f"- YOLO model: {yolo_model_path}")
        print(f"- DETR model: {detr_model_name}")
        print(f"- DEYO enabled: {use_deyo}")

    def _init_yolo(self):
        """Initialize YOLO model if not already initialized"""
        if self.yolo_model is None:
            try:
                from ultralytics import YOLO
                # Use the local model path
                local_model_path = "/Users/apple/PythonProject2/your_project/backend/models/yolov8n.pt"
                self.yolo_model = YOLO(local_model_path)
                print("YOLO model initialized successfully using local model")
            except Exception as e:
                print(f"Error initializing YOLO model: {str(e)}")
                raise

    def _init_detr(self):
        """Initialize DETR model if not already initialized"""
        if self.detr_model is None or self.detr_processor is None:
            try:
                from transformers import DetrImageProcessor, DetrForObjectDetection
                self.detr_processor = DetrImageProcessor.from_pretrained(self.detr_model_name)
                self.detr_model = DetrForObjectDetection.from_pretrained(self.detr_model_name)
                print("DETR model initialized successfully")
            except Exception as e:
                print(f"Error initializing DETR model: {str(e)}")
                raise

    def _init_deyo(self):
        """Initialize DEYO model if not already initialized"""
        if self.use_deyo and self.deyo_model is None:
            try:
                # Initialize YOLO and DETR first
                self._init_yolo()
                self._init_detr()

                # DEYO is a conceptual combination of YOLO and DETR
                # Here we just store references to both models
                self.deyo_model = {
                    "yolo": self.yolo_model,
                    "detr": (self.detr_model, self.detr_processor)
                }
                print("DEYO model initialized successfully")
            except Exception as e:
                print(f"Error initializing DEYO model: {str(e)}")
                raise

    def detect_objects_yolo(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Run YOLO object detection

        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detections

        Returns:
            List of detection dictionaries
        """
        self._init_yolo()

        try:
            results = self.yolo_model(image_path, conf=conf_threshold)

            # Process results into standardized format
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "model": "yolo"
                    })

            return detections
        except Exception as e:
            print(f"Error in YOLO detection: {str(e)}")
            return []

    def detect_objects_detr(self, image_path: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Run DETR object detection

        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for detections

        Returns:
            List of detection dictionaries
        """
        self._init_detr()

        try:
            image = Image.open(image_path)
            inputs = self.detr_processor(images=image, return_tensors="pt")
            outputs = self.detr_model(**inputs)

            # Convert outputs to same format as YOLO
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )[0]

            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class": self.detr_model.config.id2label[label.item()],
                    "confidence": score.item(),
                    "model": "detr"
                })

            return detections
        except Exception as e:
            print(f"Error in DETR detection: {str(e)}")
            return []

    def detect_objects_deyo(self, image_path: str) -> List[Dict[str, Any]]:
        """Run DEYO object detection (YOLO + DETR ensemble)

        Args:
            image_path: Path to the image file

        Returns:
            List of detection dictionaries
        """
        self._init_deyo()

        try:
            # Get detections from both models
            yolo_detections = self.detect_objects_yolo(image_path)
            detr_detections = self.detect_objects_detr(image_path)

            # Combine detections (simple approach - just concatenate)
            # A more sophisticated approach would merge overlapping boxes
            detections = yolo_detections + detr_detections

            # Mark as coming from the ensemble model
            for detection in detections:
                detection["model"] = "deyo"

            return detections
        except Exception as e:
            print(f"Error in DEYO detection: {str(e)}")
            return []

    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """Run object detection with the appropriate model

        Args:
            image_path: Path to the image file

        Returns:
            List of detection dictionaries
        """
        if self.use_deyo:
            return self.detect_objects_deyo(image_path)
        else:
            # Run both models and combine results
            yolo_detections = self.detect_objects_yolo(image_path)
            detr_detections = self.detect_objects_detr(image_path)
            return yolo_detections + detr_detections


class EnhancedFinancialImageAnalyzer:
    def __init__(self,
                 together_api_key: str,
                 vision_llm_model: str = "together/llama-3-70b-vision",
                 reasoning_llm_model: str = "Qwen/Qwen3-235B-A22B-fp8-tput",
                 text_llm_model: str = "together/llama-3-8b-instruct",
                 yolo_model_path: str = "/Users/apple/PythonProject2/your_project/backend/models/yolov8n.pt",
                 detr_model_name: str = "facebook/detr-resnet-50",
                 use_deyo: bool = False,
                 use_cache: bool = True,
                 cache_dir: str = "cache",
                 parallel_execution: bool = True,
                 max_workers: int = 3):
        """Initialize the enhanced financial image analyzer with object detection models

        Args:
            together_api_key: API key for Together AI
            vision_llm_model: Model for vision-related tasks
            reasoning_llm_model: Model for complex reasoning tasks
            text_llm_model: Model for text processing tasks
            yolo_model_path: Path to YOLO model weights
            detr_model_name: HuggingFace model name for DETR
            use_deyo: Whether to use DEYO architecture
            use_cache: Whether to cache results
            cache_dir: Directory to store cache files
            parallel_execution: Whether to run analyses in parallel
            max_workers: Maximum number of parallel workers
        """
        # Initialize the base LLMFinancialImageAnalyzer
        self.together_api_key = together_api_key
        self.vision_llm_model = vision_llm_model
        self.reasoning_llm_model = reasoning_llm_model
        self.text_llm_model = text_llm_model

        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers

        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Initialize object detection module
        self.object_detector = ObjectDetectionModule(
            yolo_model_path=yolo_model_path,
            detr_model_name=detr_model_name,
            use_deyo=use_deyo
        )

        # Define taxonomy
        self.taxonomy = self._create_taxonomy()

        print(f"Enhanced Financial Image Analyzer initialized with:")
        print(f"- Vision LLM: {vision_llm_model}")
        print(f"- Reasoning LLM: {reasoning_llm_model}")
        print(f"- Text LLM: {text_llm_model}")
        print(f"- Parallel execution: {parallel_execution}")

    def _create_taxonomy(self) -> Dict[str, List[str]]:
        """Create taxonomy of analysis categories and subcategories"""
        return {
            "Color Palette": ["Dominant colors", "Brightness", "Warm vs cool tones", "Contrast level"],
            "Layout & Composition": ["Text-to-image ratio", "Left vs right alignment", "Symmetry", "Whitespace usage"],
            "Image Type": ["Image focus type", "Visual format", "Illustration vs photo"],
            "Elements": ["Number of products shown", "Number of people shown", "Design density"],
            "Presence of Text": ["Embedded text present", "Text language", "Font style"],
            "Theme": ["Festival/special occasion logo", "Festival name", "Logo size", "Logo placement"],
            "CTA": ["Call-to-action button present", "CTA placement", "CTA contrast"],
            "Object Detection": ["Objects visible", "Brand logo visible", "Brand logo size"],
            "Character": ["Emotion (if faces shown)", "Gender shown (if people shown)"],
            "Character Role": ["Employment type (if shown)"],
            "Context": ["Environment type", "Location hints"],
            "Offer": ["Offer text present", "Offer type", "Offer text size", "Offer text position"]
        }

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 string

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_cache_path(self, image_path: str) -> str:
        """Get cache file path for an image

        Args:
            image_path: Path to the image file

        Returns:
            Path to the cache file
        """
        image_name = os.path.basename(image_path)
        cache_name = f"{os.path.splitext(image_name)[0]}_analysis.json"
        return os.path.join(self.cache_dir, cache_name)

    def _load_from_cache(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Load analysis results from cache

        Args:
            image_path: Path to the image file

        Returns:
            Cached analysis results or None if not found
        """
        cache_path = self._get_cache_path(image_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
        return None

    def _save_to_cache(self, image_path: str, results: Dict[str, Any]) -> None:
        """Save analysis results to cache

        Args:
            image_path: Path to the image file
            results: Analysis results to cache
        """
        cache_path = self._get_cache_path(image_path)
        try:
            with open(cache_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")

    def _call_together_api(self, model: str, messages: List[Dict[str, Any]],
                           max_tokens: int = 1024, temperature: float = 0.2) -> Dict[str, Any]:
        """Call Together AI API

        Args:
            model: Model name
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            API response
        """
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        # Add retry logic
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    retry_delay = min(retry_delay * 2, 60)  # Exponential backoff
                    print(f"Rate limited. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        return {"error": f"API error: {response.status_code} - {response.text}"}
            except Exception as e:
                print(f"Request error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return {"error": f"Request error: {str(e)}"}

        return {"error": "Max retries exceeded"}

    def _extract_json_from_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract JSON from API response

        Args:
            response: API response

        Returns:
            Extracted JSON data
        """
        if "error" in response:
            return {"error": response["error"]}

        try:
            content = response["choices"][0]["message"]["content"]

            # Try to extract JSON from the content
            try:
                # Check if the entire content is JSON
                return json.loads(content)
            except:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'``````', content)
                if json_match:
                    return json.loads(json_match.group(1))

                # Try to find anything that looks like JSON
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    return json.loads(json_match.group(1))

                # If all else fails, return the content as is
                return {"raw_content": content}
        except Exception as e:
            return {"error": f"Error extracting JSON: {str(e)}",
                    "raw_content": content if 'content' in locals() else "No content"}

    def _add_confidence_scores(self, result: Dict[str, Any], confidence: float = 0.9) -> Dict[str, Any]:
        """Add confidence scores to analysis results

        Args:
            result: Analysis results dictionary
            confidence: Confidence score to add (default: 0.9)

        Returns:
            Updated results dictionary with confidence scores
        """
        # Create a copy of the keys to avoid modifying during iteration
        result_keys = [key for key in list(result.keys()) if key not in ["error", "raw_content"]]

        # Add confidence scores
        for key in result_keys:
            result[f"{key}_confidence"] = confidence

        return result

    # Include all prompt methods from LLMFinancialImageAnalyzer
    def _color_palette_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for color palette analysis"""
        return [
            {
                "role": "system",
                "content": """You are a financial image analyzer. Analyze this image and provide ONLY the following specific tags:

1. Dominant colors: List the main colors (Red, Yellow, Blue, Green, Black, White, etc.)
2. Brightness: Categorize as Dark or Light
3. Warm vs cool tones: Categorize as Warm, Cool, or Neutral
4. Contrast level: Categorize as High, Medium, or Low

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the color palette of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _layout_composition_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for layout and composition analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following layout information:

1. Text-to-image ratio: Estimate as percentage (10%, 30%, 50%, etc.)
2. Left vs right alignment: Categorize as Left, Right, or Center
3. Symmetry: Categorize as Symmetrical or Asymmetrical
4. Whitespace usage: Categorize as Low, Medium, or High

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the layout and composition of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _image_type_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for image type analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following image type information:

1. Image focus type: Categorize as Product, Lifestyle, or Mixed
2. Visual format: Categorize as Static, Animated, or Video
3. Illustration vs photo: Categorize as Illustration or Photograph

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the image type of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _elements_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for elements analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following element information:

1. Number of products shown: Count as 1, 2, 3+
2. Number of people shown: Count as 0, 1, 2, 3+
3. Design density: Categorize as Minimal, Medium, or Crowded

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the elements in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _text_presence_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for text presence analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following text information:

1. Embedded text present: Answer Yes or No
2. Text language: Identify as English, Hindi, Marathi, etc.
3. Font style: Categorize as Bold, Serif, Sans-serif, or Script

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the text presence in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _theme_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for theme analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following theme information:

1. Festival/special occasion logo: Answer Yes or No
2. Festival name: Identify as Diwali, Holi, Independence Day, or None
3. Logo size: Categorize as Small, Medium, or Large
4. Logo placement: Identify as Top Left, Top Right, Bottom Left, Bottom Right, or Center

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the theme of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _cta_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for CTA analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following CTA information:

1. Call-to-action button present: Answer Yes or No
2. CTA placement: Identify as Top, Center, or Bottom
3. CTA contrast: Categorize as High, Medium, or Low

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the CTA in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _object_detection_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for object detection"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following object information:

1. Objects visible: List visible objects (TV, Sofa, Person, Phone, Refrigerator, etc.)
2. Brand logo visible: Answer Yes or No
3. Brand logo size: Categorize as Small, Medium, or Large

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify objects in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _object_verification_prompt(self, image_base64: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate prompt for verifying object detections with LLM

        Args:
            image_base64: Base64 encoded image
            detections: List of object detections from models

        Returns:
            Prompt for LLM verification
        """
        # Format detections for the prompt
        detection_text = ""
        for i, det in enumerate(detections):
            detection_text += f"{i + 1}. Class: {det['class']}, Confidence: {det['confidence']:.2f}, Model: {det['model']}\n"

        return [
            {
                "role": "system",
                "content": """You are an expert in computer vision and object detection. 
                Analyze this image and verify the detected objects. For each detection:
                1. Confirm if the object is actually present
                2. Verify if the classification is correct
                3. Suggest any missed objects
                4. Provide confidence in your assessment (0-1)

                Format your response as a JSON with these keys:
                - "verified_detections": List of objects you confirm are present
                - "corrected_detections": List of objects with corrected classifications
                - "missed_objects": List of objects that were missed
                - "confidence": Your overall confidence in this assessment
                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Verify these object detections:\n{detection_text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _character_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for character analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following character information:

1. Emotion (if faces shown): Categorize as Happy, Excited, Neutral, Angry, or None
2. Gender shown (if people shown): Categorize as Male, Female, Both, or Not Applicable

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the characters in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _character_role_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for character role analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following character role information:

1. Employment type (if shown): Identify as Doctor, Student, Businessperson, or None

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the character roles in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _context_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for context analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following context information:

1. Environment type: Categorize as Indoor, Outdoor, Office, or Natural
2. Location hints: Identify as Kitchen, Park, Store, Living Room, or None

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the context of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _offer_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        """Generate prompt for offer analysis"""
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following offer information:

1. Offer text present: Answer Yes or No
2. Offer type: Categorize as Discount, Cashback, Freebie, Combo, or None
3. Offer text size: Categorize as Small, Medium, or Large
4. Offer text position: Identify as Top Left, Top Right, Center, or Bottom

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the offers in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _integration_reasoning_prompt(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prompt for integrating and reasoning about all analysis results"""
        return [
            {
                "role": "system",
                "content": """You are an expert financial marketing analyst. Review the provided analysis results and:
1. Identify any inconsistencies or errors
2. Determine the overall marketing strategy and target audience
3. Assess the effectiveness of the visual communication

Format your response as a structured JSON."""
            },
            {
                "role": "user",
                "content": f"Integrate and reason about these analyses of a financial marketing image:\n{json.dumps(analysis_results, indent=2)}"
            }
        ]

    def _run_enhanced_object_detection(self, image_path: str) -> Dict[str, Any]:
        """Run enhanced object detection with model ensemble and LLM verification

        Args:
            image_path: Path to the image file

        Returns:
            Object detection results
        """
        try:
            # Run object detection with models
            detections = self.object_detector.detect_objects(image_path)

            if not detections:
                # Fall back to LLM-based detection if model detection fails
                return self._run_llm_object_detection(self._encode_image(image_path))

            # Verify detections with LLM
            image_base64 = self._encode_image(image_path)
            verification_prompt = self._object_verification_prompt(image_base64, detections)
            verification_response = self._call_together_api(self.vision_llm_model, verification_prompt)
            verification_result = self._extract_json_from_response(verification_response)

            # Process verified detections
            verified_objects = []

            # Add verified detections
            if "verified_detections" in verification_result:
                verified_objects.extend(verification_result["verified_detections"])

            # Add corrected detections
            if "corrected_detections" in verification_result:
                verified_objects.extend(verification_result["corrected_detections"])

            # Add missed objects
            if "missed_objects" in verification_result:
                verified_objects.extend(verification_result["missed_objects"])

            # If no verified objects, use the original detections
            if not verified_objects and detections:
                verified_objects = [det["class"] for det in detections]

            # Check for brand logos
            has_logo = any("logo" in det["class"].lower() for det in detections)
            if not has_logo and "brand logo" in str(verification_result).lower():
                has_logo = True

            # Determine logo size if present
            logo_size = "None"
            if has_logo:
                logo_detections = [det for det in detections if "logo" in det["class"].lower()]
                if logo_detections:
                    # Calculate logo size based on bounding box area relative to image size
                    img = cv2.imread(image_path)
                    img_area = img.shape[0] * img.shape[1]
                    logo_det = max(logo_detections,
                                   key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
                    logo_area = (logo_det["box"][2] - logo_det["box"][0]) * (logo_det["box"][3] - logo_det["box"][1])
                    logo_ratio = logo_area / img_area

                    if logo_ratio < 0.05:
                        logo_size = "Small"
                    elif logo_ratio < 0.15:
                        logo_size = "Medium"
                    else:
                        logo_size = "Large"
                else:
                    logo_size = "Small"  # Default if we can't calculate

            # Format results
            result = {
                "Objects visible": ", ".join(set(verified_objects)) if verified_objects else "None",
                "Brand logo visible": "Yes" if has_logo else "No",
                "Brand logo size": logo_size,
            }

            # Add confidence scores
            confidence = verification_result.get("confidence", 0.95)
            return self._add_confidence_scores(result, confidence)

        except Exception as e:
            print(f"Error in enhanced object detection: {str(e)}")
            # Fall back to LLM-based detection
            return self._run_llm_object_detection(self._encode_image(image_path))

    def _run_llm_object_detection(self, image_base64: str) -> Dict[str, Any]:
        """Run object detection using LLM only (fallback)

        Args:
            image_base64: Base64 encoded image

        Returns:
            Object detection results
        """
        prompt = self._object_detection_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_color_palette_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run color palette analysis"""
        prompt = self._color_palette_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _run_layout_composition_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run layout and composition analysis"""
        prompt = self._layout_composition_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_image_type_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run image type analysis"""
        prompt = self._image_type_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _run_elements_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run elements analysis"""
        prompt = self._elements_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_text_presence_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run text presence analysis"""
        prompt = self._text_presence_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _run_theme_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run theme analysis"""
        prompt = self._theme_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.8)

    def _run_cta_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run CTA analysis"""
        prompt = self._cta_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _run_character_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run character analysis"""
        prompt = self._character_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_character_role_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run character role analysis"""
        prompt = self._character_role_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.8)

    def _run_context_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run context analysis"""
        prompt = self._context_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_offer_analysis(self, image_base64: str) -> Dict[str, Any]:
        """Run offer analysis"""
        prompt = self._offer_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _get_default_values_for_category(self, category: str) -> Dict[str, Any]:
        """Get default values for a category"""
        defaults = {
            "Color Palette": {
                "Dominant colors": "Unknown",
                "Brightness": "Medium",
                "Warm vs cool tones": "Neutral",
                "Contrast level": "Medium",
                "Dominant colors_confidence": 0.5,
                "Brightness_confidence": 0.5,
                "Warm vs cool tones_confidence": 0.5,
                "Contrast level_confidence": 0.5
            },
            "Layout & Composition": {
                "Text-to-image ratio": "30%",
                "Left vs right alignment": "Center",
                "Symmetry": "Asymmetrical",
                "Whitespace usage": "Medium",
                "Text-to-image ratio_confidence": 0.5,
                "Left vs right alignment_confidence": 0.5,
                "Symmetry_confidence": 0.5,
                "Whitespace usage_confidence": 0.5
            },
            "Image Type": {
                "Image focus type": "Mixed",
                "Visual format": "Static",
                "Illustration vs photo": "Photograph",
                "Image focus type_confidence": 0.5,
                "Visual format_confidence": 0.5,
                "Illustration vs photo_confidence": 0.5
            },
            "Elements": {
                "Number of products shown": "1",
                "Number of people shown": "0",
                "Design density": "Medium",
                "Number of products shown_confidence": 0.5,
                "Number of people shown_confidence": 0.5,
                "Design density_confidence": 0.5
            },
            "Presence of Text": {
                "Embedded text present": "Yes",
                "Text language": "English",
                "Font style": "Sans-serif",
                "Embedded text present_confidence": 0.5,
                "Text language_confidence": 0.5,
                "Font style_confidence": 0.5
            },
            "Theme": {
                "Festival/special occasion logo": "No",
                "Festival name": "None",
                "Logo size": "None",
                "Logo placement": "None",
                "Festival/special occasion logo_confidence": 0.5,
                "Festival name_confidence": 0.5,
                "Logo size_confidence": 0.5,
                "Logo placement_confidence": 0.5
            },
            "CTA": {
                "Call-to-action button present": "No",
                "CTA placement": "None",
                "CTA contrast": "None",
                "Call-to-action button present_confidence": 0.5,
                "CTA placement_confidence": 0.5,
                "CTA contrast_confidence": 0.5
            },
            "Object Detection": {
                "Objects visible": "None",
                "Brand logo visible": "No",
                "Brand logo size": "None",
                "Objects visible_confidence": 0.5,
                "Brand logo visible_confidence": 0.5,
                "Brand logo size_confidence": 0.5
            },
            "Character": {
                "Emotion (if faces shown)": "None",
                "Gender shown (if people shown)": "Not Applicable",
                "Emotion (if faces shown)_confidence": 0.5,
                "Gender shown (if people shown)_confidence": 0.5
            },
            "Character Role": {
                "Employment type (if shown)": "None",
                "Employment type (if shown)_confidence": 0.5
            },
            "Context": {
                "Environment type": "None",
                "Location hints": "None",
                "Environment type_confidence": 0.5,
                "Location hints_confidence": 0.5
            },
            "Offer": {
                "Offer text present": "No",
                "Offer type": "None",
                "Offer text size": "None",
                "Offer text position": "None",
                "Offer text present_confidence": 0.5,
                "Offer type_confidence": 0.5,
                "Offer text size_confidence": 0.5,
                "Offer text position_confidence": 0.5
            }
        }

        return defaults.get(category, {})

    def analyze_full_image(self, image_path: str) -> Dict[str, Dict[str, Any]]:
        """Analyze an image using multiple specialized LLM agents

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with analysis results
        """
        # Check cache if enabled
        if self.use_cache:
            cached_results = self._load_from_cache(image_path)
            if cached_results is not None:
                return cached_results

        # Get raw LLM analysis results
        results = self.analyze_image(image_path)

        # Save to cache if enabled
        if self.use_cache:
            self._save_to_cache(image_path, results)

        return results

    def focused_analysis(self, image_path: str, categories: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze an image for specific categories

        Args:
            image_path: Path to the image file
            categories: List of categories to analyze

        Returns:
            Dictionary with analysis results for requested categories
        """
        # Analyze the image with only the requested categories
        results = self.analyze_image(image_path, categories=categories)

        # Filter to include only requested categories
        return {k: v for k, v in results.items() if k in categories}

    def analyze_image(self, image_path: str, categories: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Analyze an image using multiple specialized LLM agents

        Args:
            image_path: Path to the image file
            categories: Specific categories to analyze (None for all)

        Returns:
            Dictionary with analysis results
        """
        # Convert image to base64
        image_base64 = self._encode_image(image_path)

        # Determine which analyses to run
        all_categories = list(self.taxonomy.keys())
        categories_to_analyze = categories if categories is not None else all_categories

        # Run specialized analyses
        analysis_results = {}

        if self.parallel_execution:
            # Run analyses in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}

                # Create all futures first
                for category in categories_to_analyze:
                    if category == "Color Palette":
                        futures[category] = executor.submit(self._run_color_palette_analysis, image_base64)
                    elif category == "Layout & Composition":
                        futures[category] = executor.submit(self._run_layout_composition_analysis, image_base64)
                    elif category == "Image Type":
                        futures[category] = executor.submit(self._run_image_type_analysis, image_base64)
                    elif category == "Elements":
                        futures[category] = executor.submit(self._run_elements_analysis, image_base64)
                    elif category == "Presence of Text":
                        futures[category] = executor.submit(self._run_text_presence_analysis, image_base64)
                    elif category == "Theme":
                        futures[category] = executor.submit(self._run_theme_analysis, image_base64)
                    elif category == "CTA":
                        futures[category] = executor.submit(self._run_cta_analysis, image_base64)
                    elif category == "Object Detection":
                        futures[category] = executor.submit(self._run_enhanced_object_detection, image_path)
                    elif category == "Character":
                        futures[category] = executor.submit(self._run_character_analysis, image_base64)
                    elif category == "Character Role":
                        futures[category] = executor.submit(self._run_character_role_analysis, image_base64)
                    elif category == "Context":
                        futures[category] = executor.submit(self._run_context_analysis, image_base64)
                    elif category == "Offer":
                        futures[category] = executor.submit(self._run_offer_analysis, image_base64)

                # Then collect results
                for category, future in futures.items():
                    try:
                        analysis_results[category] = future.result()
                    except Exception as e:
                        print(f"Error in {category} analysis: {str(e)}")
                        analysis_results[category] = self._get_default_values_for_category(category)
        else:
            # Run analyses sequentially
            for category in categories_to_analyze:
                try:
                    if category == "Color Palette":
                        analysis_results[category] = self._run_color_palette_analysis(image_base64)
                    elif category == "Layout & Composition":
                        analysis_results[category] = self._run_layout_composition_analysis(image_base64)
                    elif category == "Image Type":
                        analysis_results[category] = self._run_image_type_analysis(image_base64)
                    elif category == "Elements":
                        analysis_results[category] = self._run_elements_analysis(image_base64)
                    elif category == "Presence of Text":
                        analysis_results[category] = self._run_text_presence_analysis(image_base64)
                    elif category == "Theme":
                        analysis_results[category] = self._run_theme_analysis(image_base64)
                    elif category == "CTA":
                        analysis_results[category] = self._run_cta_analysis(image_base64)
                    elif category == "Object Detection":
                        analysis_results[category] = self._run_enhanced_object_detection(image_path)
                    elif category == "Character":
                        analysis_results[category] = self._run_character_analysis(image_base64)
                    elif category == "Character Role":
                        analysis_results[category] = self._run_character_role_analysis(image_base64)
                    elif category == "Context":
                        analysis_results[category] = self._run_context_analysis(image_base64)
                    elif category == "Offer":
                        analysis_results[category] = self._run_offer_analysis(image_base64)
                except Exception as e:
                    print(f"Error in {category} analysis: {str(e)}")
                    analysis_results[category] = self._get_default_values_for_category(category)

        # Add any missing categories with default values
        for category in categories_to_analyze:
            if category not in analysis_results:
                analysis_results[category] = self._get_default_values_for_category(category)

        return analysis_results

    def compare_images(self, image_paths: List[str], categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple images

        Args:
            image_paths: List of paths to image files
            categories: Specific categories to compare (None for all)

        Returns:
            Comparison results
        """
        # Analyze all images
        analyses = {}
        for image_path in image_paths:
            try:
                analyses[image_path] = self.analyze_image(image_path, categories)
            except Exception as e:
                print(f"Error analyzing {image_path}: {str(e)}")
                analyses[image_path] = {"error": str(e)}

        # Create comparison prompt
        comparison_prompt = self._integration_reasoning_prompt(analyses)

        # Get comparison results
        response = self._call_together_api(self.reasoning_llm_model, comparison_prompt, max_tokens=2048)
        comparison = self._extract_json_from_response(response)

        return {
            "individual_analyses": analyses,
            "comparison": comparison
        }

    def visualize_detections(self, image_path: str, output_path: Optional[str] = None) -> str:
        """Visualize object detections on an image

        Args:
            image_path: Path to the image file
            output_path: Path to save the output image (default: adds "_detected" suffix)

        Returns:
            Path to the output image
        """
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_detected.jpg"

        # Run object detection
        detections = self.object_detector.detect_objects(image_path)

        # Load image
        img = cv2.imread(image_path)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det["box"]]
            class_name = det["class"]
            confidence = det["confidence"]
            model_name = det["model"]

            # Draw bounding box
            color = (0, 255, 0)  # Green for YOLO
            if model_name == "detr":
                color = (255, 0, 0)  # Blue for DETR
            elif model_name == "deyo":
                color = (0, 0, 255)  # Red for DEYO

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save image
        cv2.imwrite(output_path, img)

        return output_path

