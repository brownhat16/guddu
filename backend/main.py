from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
from typing import Dict, Any, List
import logging
import time
from dotenv import load_dotenv
import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Load environment variables
load_dotenv()

# Import our enhanced image analyzer
from tempfinancial import EnhancedFinancialImageAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("financial-image-analyzer")

app = FastAPI(
    title="Enhanced Financial Image Analyzer API",
    description="API for analyzing financial marketing images with YOLO, DETR object detection and LLM analysis",
    version="1.1.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create directories for temporary uploads and visualizations
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# Get API key from environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "ceeabe5b322559bd0fe8a21e18df89860cf31aa2af6e4b31cc60acd299f2d9c0")

# Initialize the analyzer when the app starts
logger.info("Initializing the EnhancedFinancialImageAnalyzer. This may take a few moments...")
analyzer = EnhancedFinancialImageAnalyzer(
    together_api_key=TOGETHER_API_KEY,
    vision_llm_model="Qwen/Qwen2.5-VL-72B-Instruct",
    reasoning_llm_model="Qwen/Qwen3-235B-A22B-fp8-tput",
    text_llm_model="meta-llama/Meta-Llama-Guard-3-8B",
    yolo_model_path="/Users/apple/PythonProject2/your_project/backend/models/yolov8n.pt",  # Using the latest YOLOv9
    detr_model_name="facebook/detr-resnet-50",
    use_deyo=False,  # Not using DEYO for now
    use_cache=True,
    cache_dir="cache",
    parallel_execution=True,
    max_workers=3
)
logger.info("Enhanced Image analyzer initialized successfully!")


@app.get("/")
def read_root():
    """Basic health check endpoint"""
    return {
        "status": "ok",
        "message": "Enhanced Financial Image Analyzer API is ready",
        "version": "1.1.0",
        "models": {
            "vision_llm": analyzer.vision_llm_model,
            "reasoning_llm": analyzer.reasoning_llm_model,
            "yolo": analyzer.object_detector.yolo_model_path,
            "detr": analyzer.object_detector.detr_model_name
        }
    }


@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze a financial marketing image and return tagged attributes

    - Accepts image uploads (.jpg, .jpeg, .png)
    - Returns categorized tags without confidence scores
    """
    start_time = time.time()

    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a JPG, PNG, or BMP image."
        )

    # Create a unique filename for the temporary file
    temp_file_path = f"temp_uploads/temp_{int(time.time())}_{file.filename}"

    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing image: {file.filename}")

        # Analyze the image with the enhanced analyzer
        full_results = analyzer.analyze_image(temp_file_path)

        # Filter out confidence scores
        tags_only_results = {}
        for category, values in full_results.items():
            tags_only_results[category] = {k: v for k, v in values.items() if not k.endswith("_confidence")}

        processing_time = time.time() - start_time
        logger.info(f"Image processed in {processing_time:.2f} seconds")

        return JSONResponse(content={
            "results": tags_only_results,
            "filename": file.filename,
            "processing_time_seconds": round(processing_time, 2)
        })

    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/analyze-image/focused/")
async def focused_analyze_image(
        file: UploadFile = File(...),
        categories: str = "Color Palette,Layout & Composition,Object Detection"
) -> Dict[str, Any]:
    """
    Analyze a financial marketing image with focus on specific categories

    - Accepts image uploads (.jpg, .jpeg, .png)
    - Allows specifying categories to analyze (comma-separated)
    - Returns only the specified categories of tags without confidence scores
    """
    start_time = time.time()

    # Parse categories from comma-separated string
    category_list = [cat.strip() for cat in categories.split(",")]

    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a JPG, PNG, or BMP image."
        )

    # Create a unique filename for the temporary file
    temp_file_path = f"temp_uploads/temp_{int(time.time())}_{file.filename}"

    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing image with focus on: {categories}")

        # Perform focused analysis with the enhanced analyzer
        focused_results = analyzer.analyze_image(temp_file_path, categories=category_list)

        # Filter out confidence scores
        tags_only_results = {}
        for category, values in focused_results.items():
            tags_only_results[category] = {k: v for k, v in values.items() if not k.endswith("_confidence")}

        processing_time = time.time() - start_time
        logger.info(f"Focused analysis completed in {processing_time:.2f} seconds")

        return JSONResponse(content={
            "results": tags_only_results,
            "filename": file.filename,
            "categories_analyzed": category_list,
            "processing_time_seconds": round(processing_time, 2)
        })

    except Exception as e:
        logger.error(f"Error in focused analysis of {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/visualize-detections/")
async def visualize_detections(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Visualize object detections on an image

    - Accepts image uploads (.jpg, .jpeg, .png)
    - Returns a URL to the visualization image
    """
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a JPG, PNG, or BMP image."
        )

    # Create unique filenames for the temporary and output files
    timestamp = int(time.time())
    temp_file_path = f"temp_uploads/temp_{timestamp}_{file.filename}"
    output_filename = f"detected_{timestamp}_{file.filename}"
    output_path = f"visualizations/{output_filename}"

    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Visualizing detections for: {file.filename}")

        # Generate visualization
        visualization_path = analyzer.visualize_detections(temp_file_path, output_path)

        # Generate URL for the visualization
        # In a real deployment, you would use your domain name
        visualization_url = f"/visualizations/{os.path.basename(visualization_path)}"

        return JSONResponse(content={
            "filename": file.filename,
            "visualization_path": visualization_url
        })

    except Exception as e:
        logger.error(f"Error visualizing detections for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error visualizing detections: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/compare-images/")
async def compare_images(files: List[UploadFile] = File(...), categories: str = None) -> Dict[str, Any]:
    """
    Compare multiple financial marketing images

    - Accepts multiple image uploads
    - Optionally allows specifying categories to compare (comma-separated)
    - Returns comparison results
    """
    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least two images to compare"
        )

    # Parse categories if provided
    category_list = None
    if categories:
        category_list = [cat.strip() for cat in categories.split(",")]

    temp_file_paths = []
    try:
        # Save all uploaded files
        for file in files:
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file format for {file.filename}. Please upload only image files."
                )

            temp_path = f"temp_uploads/temp_{int(time.time())}_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_file_paths.append(temp_path)

        logger.info(f"Comparing {len(files)} images")

        # Compare the images
        comparison_results = analyzer.compare_images(temp_file_paths, categories=category_list)

        # Filter out confidence scores from individual analyses
        filtered_analyses = {}
        for path, analysis in comparison_results["individual_analyses"].items():
            filename = os.path.basename(path)
            filtered_analyses[filename] = {}
            for category, values in analysis.items():
                if category != "error":
                    filtered_analyses[filename][category] = {
                        k: v for k, v in values.items() if not k.endswith("_confidence")
                    }
                else:
                    filtered_analyses[filename][category] = values

        return JSONResponse(content={
            "individual_analyses": filtered_analyses,
            "comparison": comparison_results["comparison"],
            "filenames": [file.filename for file in files]
        })

    except Exception as e:
        logger.error(f"Error comparing images: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")
    finally:
        # Clean up all temporary files
        for path in temp_file_paths:
            if os.path.exists(path):
                os.remove(path)


import os
import uvicorn

# Your FastAPI app code

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

