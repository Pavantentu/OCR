import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import sys
import uuid
import shutil
import traceback
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
import base64
from io import BytesIO
from PIL import Image
import requests
import pandas as pd
import copy
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
from functools import lru_cache
import threading

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fastapi_app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# TrOCR imports
try:
    import torch
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    TROCR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TrOCR dependencies not available: {e}. Install with: pip install transformers torch")
    TROCR_AVAILABLE = False
    torch = None
    VisionEncoderDecoderModel = None
    TrOCRProcessor = None

# Import your validation and detection modules
# Assuming these are in the same directory and don't rely on Flask global state
from validate import process_with_validation, SHGImageValidator
from test import SHGFormDetector

# SHG table layout definitions (same as app.py)
SHG_COLUMN_LAYOUT = [
    {"key": "member_mbk_id", "label": "సభ్యురాలి MBK ID"},
    {"key": "member_name", "label": "సభ్యురాలు పేరు"},
    {"key": "savings_this_month", "label": "ఈ నెల పొదుపు"},
    {"key": "savings_till_now", "label": "ఈ నెల వరకు పొదుపు"},
    {"key": "shg_internal_loan_total", "label": "SHG అంతర్గత అప్పు కట్టిన మొత్తం"},
    {"key": "bank_loan_total", "label": "బ్యాంక్ అప్పు కట్టిన మొత్తం"},
    {"key": "streenidhi_micro_loan_total", "label": "స్త్రీనిధి మైక్రో అప్పు కట్టిన మొత్తం"},
    {"key": "streenidhi_tenni_loan_total", "label": "స్త్రీనిధి టెన్నీ అప్పు కట్టిన మొత్తం"},
    {"key": "unnathi_scsp_loan_total", "label": "ఉన్నతి (SCSP) అప్పు కట్టిన మొత్తం"},
    {"key": "unnathi_tsp_loan_total", "label": "ఉన్నతి (TSP) అప్పు కట్టిన మొత్తం"},
    {"key": "cif_loan_total", "label": "CIF అప్పు కట్టిన మొత్తం"},
    {"key": "vo_internal_loan_total", "label": "VO అంతర్గత అప్పు కట్టిన మొత్తం"},
    {"key": "loan_type", "label": "అప్పు రకం"},
    {"key": "loan_type_amount", "label": "మొత్తం"},
    {"key": "penalty_amount", "label": "జరిమానా"},
    {"key": "returned_to_members", "label": "సభ్యులకు తిరిగి ఇచ్చిన మొత్తం"},
    {"key": "other_savings_total", "label": "సభ్యుల ఇతర పొదుపులు (SN+VO+Other Saving)"},
]

BASE_SHG_HEADER_ROWS = [
    [
        {"label": "………......................... స్వయం స్హయక స్ంఘ  ................. తేదిన జరిగిన స్మావేశ ఆరిిక లావాదేవీలు వివరములు (అనుభందం - II)", "col_span": 17, "row_span": 1},
    ],
    [
        {"label": "సభ్యురాలి MBK ID", "col_span": 1, "row_span": 3},
        {"label": "సభ్యురాలు పేరు", "col_span": 1, "row_span": 3},
        {"label": "పొదుపు", "col_span": 2, "row_span": 1},
        {"label": "సభ్యుల స్థాయిలో జరిగిన ఆర్థిక లావాదేవీలు", "col_span": 8, "row_span": 1},
        {"label": "కోత్త రుణం", "col_span": 2, "row_span": 1},
        {"label": "జరిమానా", "col_span": 1, "row_span": 3},
        {"label": "సభ్యులకు తిరిగి ఇచ్చిన మొత్తం", "col_span": 1, "row_span": 3},
        {"label": "సభ్యుల ఇతర పొదుపులు (SN+VO+Other Saving)", "col_span": 1, "row_span": 3},
    ],
    [
        {"label": "ఈ నెల పొదుపు", "col_span": 1, "row_span": 2},
        {"label": "ఈ నెల వరకు పొదుపు", "col_span": 1, "row_span": 2},
        {"label": "అంపు రికార్డు వివరాలు", "col_span": 8, "row_span": 1},
        {"label": "అప్పు రకం", "col_span": 1, "row_span": 2},
        {"label": "మొత్తం", "col_span": 1, "row_span": 2},
    ],
    [
        {"label": "SHG అంతర్గత అప్పు కట్టిన మొత్తం", "col_span": 1, "row_span": 1},
        {"label": "బ్యాంక్ అప్పు కట్టిన మొత్తం", "col_span": 1, "row_span": 1},
        {"label": "స్త్రీనిధి మైక్రో అప్పు కట్టిన మొత్తం", "col_span": 1, "row_span": 1},
        {"label": "స్త్రీనిధి టెన్నీ అప్పు కట్టిన మొత్తం", "col_span": 1, "row_span": 1},
        {"label": "ఉన్నతి (SCSP) అప్పు కట్టిన మొత్తం", "col_span": 1, "row_span": 1},
        {"label": "ఉన్నతి (TSP) అప్పు కట్టిన మొత్తం", "col_span": 1, "row_span": 1},
        {"label": "CIF అప్పు కట్టిన మొత్తం", "col_span": 1, "row_span": 1},
        {"label": "VO అంతర్గత అప్పు కట్టిన మొత్తం", "col_span": 1, "row_span": 1},
    ],
]

DEFAULT_SHG_COLUMN_COUNT = len(SHG_COLUMN_LAYOUT)

def build_shg_header_rows(shg_mbk_id: str) -> List[List[Dict[str, Any]]]:
    header_rows = copy.deepcopy(BASE_SHG_HEADER_ROWS)
    helper_row = [
        {"label": "SHG MBK ID", "col_span": 1, "row_span": 1, "align": "left"},
        {
            "label": shg_mbk_id or "",
            "col_span": 16,
            "row_span": 1,
            "align": "left",
        },
    ]
    header_rows.insert(1, helper_row)
    return header_rows

def build_shg_column_headers(total_columns: int) -> List[Dict[str, Any]]:
    total = max(total_columns, DEFAULT_SHG_COLUMN_COUNT)
    headers = []
    for idx in range(total):
        if idx < len(SHG_COLUMN_LAYOUT):
            config = SHG_COLUMN_LAYOUT[idx]
            headers.append({
                "index": idx,
                "key": config["key"],
                "label": config["label"]
            })
        else:
            headers.append({
                "index": idx,
                "key": f"column_{idx + 1}",
                "label": f"Column {idx + 1}"
            })
    return headers

from contextlib import asynccontextmanager

# TrOCR Model Config
TROCR_MODEL_PATH = Path(__file__).resolve().parent / 'trocr_model'
TROCR_BATCH_SIZE_GPU = 16  # Moderate batch size for GPU processing
TROCR_BATCH_SIZE_CPU = 1   # Process 1 at a time on CPU for responsiveness
TROCR_MAX_LENGTH = 128     # Maximum text length for recognition
MAX_CONCURRENT_BATCHES_GPU = 1  # Process batches sequentially on GPU
MAX_CONCURRENT_BATCHES_CPU = 4  # Process up to 4 single images concurrently on CPU

# Global shutdown event
shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize OCRProcessor
    logger.info("Starting up: Initializing OCRProcessor...")
    app.state.ocr_processor = OCRProcessor(debug=False)
    yield
    # Shutdown: Set shutdown event and clean up resources
    logger.info("Shutting down: Setting shutdown flag...")
    shutdown_event.set()
    logger.info("Shutting down: Cleaning up OCRProcessor resources...")
    if hasattr(app.state.ocr_processor, 'ocr_client') and hasattr(app.state.ocr_processor.ocr_client, 'cleanup'):
        app.state.ocr_processor.ocr_client.cleanup()

# FastAPI App
app = FastAPI(lifespan=lifespan)

# CORS Configuration - can be overridden via environment variables
import os
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173,https://pavantentu.github.io').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# Configuration
TEMP_FOLDER = Path('temp_processing')
UPLOAD_FOLDER = Path('uploads')
RESULT_FOLDER = Path('result')
FINANCIAL_FOLDER = Path(__file__).resolve().parent / 'financial_data'
ANALYTICS_FOLDER = Path(__file__).resolve().parent / 'analytics_data'

# Security Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf', '.xlsx', '.xls'}

for folder in [TEMP_FOLDER, UPLOAD_FOLDER, RESULT_FOLDER, FINANCIAL_FOLDER, ANALYTICS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# PDF Support
try:
    import fitz
    PDF_SUPPORT = True
    logger.info("PDF support enabled (PyMuPDF)")
except ImportError as e:
    PDF_SUPPORT = False
    logger.warning(f"PDF support disabled - install PyMuPDF: pip install PyMuPDF. Error: {e}")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    # Remove any path components
    filename = os.path.basename(filename)
    # Remove any potentially dangerous characters
    filename = filename.replace('..', '')
    # Keep only alphanumeric, dots, hyphens, and underscores
    import re
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename

def validate_file_upload(filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
    """Validate file upload for security.
    
    Args:
        filename: Name of the uploaded file
        file_size: Size of the file in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file size
    if file_size > MAX_FILE_SIZE:
        return False, f"File size exceeds maximum limit of {MAX_FILE_SIZE / (1024*1024):.0f}MB"
    
    # Sanitize and check extension
    sanitized_name = sanitize_filename(filename)
    if not sanitized_name:
        return False, "Invalid filename"
    
    ext = Path(sanitized_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type {ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, None

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    else:
        return obj

class FileProcessor:
    @staticmethod
    def is_pdf(filename: str) -> bool:
        return filename.lower().endswith('.pdf')
    
    @staticmethod
    def is_image(filename: str) -> bool:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return Path(filename).suffix.lower() in valid_extensions
    
    @staticmethod
    def pdf_to_images(pdf_path: str, output_folder: Path) -> List[Path]:
        if not PDF_SUPPORT:
            raise Exception("PDF support not available - install PyMuPDF")
        
        logger.info(f"Converting PDF to images: {pdf_path}")
        image_paths = []
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"PDF has {len(doc)} pages")
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = fitz.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_path = output_folder / f"page_{page_num + 1}.jpg"
                img.save(str(img_path), 'JPEG', quality=95)
                image_paths.append(img_path)
                logger.info(f"  Page {page_num + 1} converted: {img_path}")
            doc.close()
            return image_paths
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    
    @staticmethod
    def validate_and_convert_image(file_path: str, output_folder: Path) -> Optional[Path]:
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                logger.error(f"Could not read image: {file_path}")
                return None
            output_path = output_folder / f"{Path(file_path).stem}_converted.jpg"
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Image validated: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return None

class TrOCROCR:
    """TrOCR-based OCR using local model for text recognition"""
    
    def __init__(self, model_path: Path = None):
        self.model_path = model_path or TROCR_MODEL_PATH
        self.model = None
        self.processor = None
        self.device = None
        self._lock = threading.Lock()
        self._initialized = False
        
        if not TROCR_AVAILABLE:
            logger.error("TrOCR dependencies not available. Install with: pip install transformers torch")
            return
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize TrOCR model and processor"""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                if not self.model_path.exists():
                    logger.error(f"TrOCR model path does not exist: {self.model_path}")
                    return
                
                # Determine device (GPU if available, else CPU)
                if torch and torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = torch.device("cpu")
                    logger.info("✓ Using CPU")
                
                # Load processor
                logger.info(f"Loading TrOCR processor from {self.model_path}...")
                self.processor = TrOCRProcessor.from_pretrained(str(self.model_path))
                
                # Load model
                logger.info(f"Loading TrOCR model from {self.model_path}...")
                self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_path))
                self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
                
                # Enable half precision for faster inference on GPU
                if self.device.type == "cuda" and torch.cuda.is_available():
                    try:
                        self.model = self.model.half()  # Use FP16 for faster inference
                        logger.info("✓ Model loaded with FP16 precision for faster GPU inference")
                    except Exception as e:
                        logger.warning(f"Could not enable FP16: {e}, using FP32")
                
                self._initialized = True
                logger.info("✓ TrOCR model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize TrOCR model: {e}")
                logger.error(traceback.format_exc())
                self.model = None
                self.processor = None
    
    def is_configured(self) -> bool:
        """Check if model is loaded and ready"""
        return self._initialized and self.model is not None and self.processor is not None
    
    def _preprocess_image(self, image: np.ndarray) -> Optional[Image.Image]:
        """Convert numpy array to PIL Image, ensuring RGB format"""
        if image is None:
            return None
        try:
            # Handle grayscale images (2D arrays)
            if len(image.shape) == 2:
                # Convert grayscale to RGB by stacking the same channel 3 times
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Handle BGR images (3D arrays with 3 channels)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Handle RGBA images (3D arrays with 4 channels)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            # Already in correct format or unknown format
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            # Ensure image is in RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return pil_image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def recognize_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognize text from a single image"""
        if not self.is_configured():
            return "", 0.0
        
        pil_image = self._preprocess_image(image)
        if pil_image is None:
            return "", 0.0
        
        try:
            # Process image
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=TROCR_MAX_LENGTH,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode text
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text = text.strip()
            
            # Calculate pseudo-confidence based on text length and model output
            confidence = min(0.95, 0.5 + len(text) * 0.01) if text else 0.0
            
            return text, float(confidence)
            
        except Exception as e:
            logger.error(f"Error during TrOCR recognition: {e}")
            logger.error(traceback.format_exc())
            return "", 0.0
    
    def batch_recognize(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Batch recognition with adaptive batch sizing based on hardware"""
        if not images:
            return []
        
        if not self.is_configured():
            return [("", 0.0)] * len(images)
        
        # Determine optimal batch size based on hardware
        is_gpu = self.device.type == "cuda" if self.device else False
        batch_size = TROCR_BATCH_SIZE_GPU if is_gpu else TROCR_BATCH_SIZE_CPU
        
        logger.info(f"Processing {len(images)} images on {self.device.type.upper()} with batch size {batch_size}")
        
        # Preprocess all images and track original indices
        pil_images = []
        image_to_original_idx = []  # Maps pil_images index to original images index
        
        for idx, image in enumerate(images):
            pil_image = self._preprocess_image(image)
            if pil_image is not None:
                pil_images.append(pil_image)
                image_to_original_idx.append(idx)
        
        if not pil_images:
            return [("", 0.0)] * len(images)
        
        results = [("", 0.0)] * len(images)
        
        try:
            # Process in batches for optimal hardware utilization
            total_batches = (len(pil_images) + batch_size - 1) // batch_size
            logger.info(f"Splitting into {total_batches} batches")
            
            for batch_num, batch_start in enumerate(range(0, len(pil_images), batch_size), 1):
                batch_end = min(batch_start + batch_size, len(pil_images))
                batch_images = pil_images[batch_start:batch_end]
                batch_original_indices = image_to_original_idx[batch_start:batch_end]
                
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch_images)} images)")
                
                # Process batch
                try:
                    # Prepare pixel values for batch
                    pixel_values = self.processor(batch_images, return_tensors="pt", padding=True).pixel_values
                    pixel_values = pixel_values.to(self.device)
                    
                    # Generate text for batch
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            pixel_values,
                            max_length=TROCR_MAX_LENGTH,
                            num_beams=5,
                            early_stopping=True
                        )
                    
                    # Decode batch results
                    batch_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Store results using original indices
                    for original_idx, text in zip(batch_original_indices, batch_texts):
                        text = text.strip()
                        confidence = min(0.95, 0.5 + len(text) * 0.01) if text else 0.0
                        results[original_idx] = (text, float(confidence))
                    
                    logger.debug(f"Batch {batch_num}/{total_batches} completed successfully")
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}/{total_batches} ({batch_start}-{batch_end}): {e}")
                    # Fallback to individual processing for this batch
                    for original_idx in batch_original_indices:
                        try:
                            text, confidence = self.recognize_text(images[original_idx])
                            results[original_idx] = (text, confidence)
                        except Exception:
                            results[original_idx] = ("", 0.0)
        
        except Exception as e:
            logger.error(f"Error during batch recognition: {e}")
            logger.error(traceback.format_exc())
            # Fallback to individual processing
            for idx, image in enumerate(images):
                if results[idx] == ("", 0.0):
                    try:
                        results[idx] = self.recognize_text(image)
                    except Exception:
                        results[idx] = ("", 0.0)
        
        logger.info(f"Completed processing {len(images)} images")
        return results
    
    async def batch_recognize_async(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Async batch recognition with concurrent processing for better CPU performance"""
        if not images:
            return []
        
        if not self.is_configured():
            return [("", 0.0)] * len(images)
        
        # Determine optimal batch size and concurrency based on hardware
        is_gpu = self.device.type == "cuda" if self.device else False
        batch_size = TROCR_BATCH_SIZE_GPU if is_gpu else TROCR_BATCH_SIZE_CPU
        max_concurrent = MAX_CONCURRENT_BATCHES_GPU if is_gpu else MAX_CONCURRENT_BATCHES_CPU
        
        logger.info(f"Processing {len(images)} images on {self.device.type.upper()} with batch size {batch_size}, max concurrent batches: {max_concurrent}")
        
        # Preprocess all images
        pil_images = []
        image_to_original_idx = []
        
        for idx, image in enumerate(images):
            pil_image = self._preprocess_image(image)
            if pil_image is not None:
                pil_images.append(pil_image)
                image_to_original_idx.append(idx)
        
        if not pil_images:
            return [("", 0.0)] * len(images)
        
        results = [("", 0.0)] * len(images)
        
        # Create batches
        batches = []
        total_batches = (len(pil_images) + batch_size - 1) // batch_size
        logger.info(f"Splitting into {total_batches} batches")
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(pil_images))
            batch_images = pil_images[batch_start:batch_end]
            batch_original_indices = image_to_original_idx[batch_start:batch_end]
            batches.append((batch_num + 1, batch_images, batch_original_indices))
        
        # Process batches with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch_with_semaphore(batch_info):
            batch_num, batch_images, batch_original_indices = batch_info
            
            # Check for shutdown
            if shutdown_event.is_set():
                logger.warning(f"Shutdown detected, skipping batch {batch_num}/{total_batches}")
                return
            
            async with semaphore:
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch_images)} images)")
                
                # Run batch recognition in executor
                loop = asyncio.get_running_loop()
                try:
                    # Process batch synchronously in thread pool
                    def sync_process():
                        if shutdown_event.is_set():
                            return []
                        
                        try:
                            pixel_values = self.processor(batch_images, return_tensors="pt", padding=True).pixel_values
                            pixel_values = pixel_values.to(self.device)
                            
                            with torch.no_grad():
                                generated_ids = self.model.generate(
                                    pixel_values,
                                    max_length=TROCR_MAX_LENGTH,
                                    num_beams=5,
                                    early_stopping=True
                                )
                            
                            batch_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                            return batch_texts
                        except Exception as e:
                            logger.error(f"Error in sync_process: {e}")
                            return []
                    
                    batch_texts = await loop.run_in_executor(None, sync_process)
                    
                    if batch_texts:
                        for original_idx, text in zip(batch_original_indices, batch_texts):
                            text = text.strip()
                            confidence = min(0.95, 0.5 + len(text) * 0.01) if text else 0.0
                            results[original_idx] = (text, float(confidence))
                        logger.debug(f"Batch {batch_num}/{total_batches} completed successfully")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}/{total_batches}: {e}")
                    # Fallback to empty results for this batch
                    for original_idx in batch_original_indices:
                        if results[original_idx] == ("", 0.0):
                            results[original_idx] = ("", 0.0)
        
        # Process all batches concurrently with semaphore limit
        await asyncio.gather(*[process_batch_with_semaphore(batch) for batch in batches])
        
        logger.info(f"Completed processing {len(images)} images")
        return results
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._initialized = False
        logger.info("TrOCR model resources cleaned up")

class OCRProcessor:
    # Shared detector instance to avoid repeated initialization
    _shared_detector = None
    _detector_lock = threading.Lock()
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.validator = SHGImageValidator(debug=False)
        self.ocr_client = TrOCROCR()
        if self.ocr_client.is_configured():
            logger.info("✓ OCR Processor initialized with TrOCR")
        else:
            logger.warning("⚠ TrOCR model not loaded — OCR will return empty text until model is available")
    
    @classmethod
    def get_shared_detector(cls):
        """Get or create shared detector instance for reuse"""
        if cls._shared_detector is None:
            with cls._detector_lock:
                if cls._shared_detector is None:
                    cls._shared_detector = SHGFormDetector(
                        debug=False,
                        return_images=False,
                        intersection_scale=None
                    )
        return cls._shared_detector
    
    def process_single_image(self, image_path: str) -> Dict:
        logger.info(f"Processing image: {image_path}")
        try:
            result = process_with_validation(
                image_path,
                debug=False,
                training_mode=False,
                return_images=False
            )
            if not result or not result.get('success'):
                return {
                    'success': False,
                    'error': result.get('error', 'Validation failed') if result else 'Processing failed',
                    'validation': result.get('validation') if result else None,
                    'image_path': str(image_path)
                }
            cells = result.get('cells', [])
            shg_id = result.get('shg_id')
            logger.info(f"✓ Successfully processed: {len(cells)} cells extracted")
            return {
                'success': True,
                'image_path': str(image_path),
                'validation': result.get('validation'),
                'cells': cells,
                'shg_id': shg_id,
                'processing_summary': result.get('processing_summary'),
                'total_cells': len(cells)
            }
        except Exception as e:
            logger.error(f"Processing error: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'image_path': str(image_path)
            }
    
    def structure_shg_table_data(self, cells: List[Dict]) -> Dict:
        shg_mbk_id = ""
        for cell in cells:
            if cell.get('debug_id') == 2:
                shg_mbk_id = cell.get('text', '')
                break
        
        rows_by_index: Dict[int, Dict[int, Dict]] = {}
        max_col_index = -1
        has_row_col_metadata = False
        
        for cell in cells:
            if cell.get('debug_id') == 2:
                continue
            row_idx = cell.get('row')
            col_idx = cell.get('col')
            if row_idx is None or col_idx is None or row_idx < 0 or col_idx < 0:
                continue
            has_row_col_metadata = True
            row_idx = int(row_idx)
            col_idx = int(col_idx)
            max_col_index = max(max_col_index, col_idx)
            if row_idx not in rows_by_index:
                rows_by_index[row_idx] = {}
            rows_by_index[row_idx][col_idx] = {
                'text': cell.get('text', ''),
                'confidence': float(cell.get('confidence', 0.0)),
                'debug_id': cell.get('debug_id'),
                'bbox': cell.get('bbox')
            }
        
        if not has_row_col_metadata:
            logger.warning("Cell metadata lacks row/col indices - falling back to legacy table layout")
            return self._structure_with_legacy_layout(cells, shg_mbk_id)
        
        total_columns = max(max_col_index + 1, DEFAULT_SHG_COLUMN_COUNT)
        headers = build_shg_column_headers(total_columns)
        header_rows = build_shg_header_rows(shg_mbk_id)
        header_lookup = {header['index']: header for header in headers}
        
        data_rows = []
        sorted_rows = sorted(rows_by_index.keys())
        
        for display_idx, row_idx in enumerate(sorted_rows, start=1):
            col_map = rows_by_index[row_idx]
            row_cells_payload = []
            for header in headers:
                col_index = header['index']
                cell_data = col_map.get(col_index)
                row_cells_payload.append({
                    "col_index": col_index,
                    "key": header['key'],
                    "label": header['label'],
                    "text": cell_data.get('text', '') if cell_data else '',
                    "confidence": float(cell_data.get('confidence', 0.0)) if cell_data else 0.0,
                    "debug_id": cell_data.get('debug_id') if cell_data else None
                })
            row_payload = {
                "row_number": display_idx,
                "row_index": int(row_idx),
                "cells": row_cells_payload
            }
            legacy_keys = [
                ("member_id", 0),
                ("member_name", 1),
                ("this_month_savings", 2),
                ("total_savings", 3),
            ]
            for legacy_key, col_index in legacy_keys:
                if col_index < len(row_cells_payload):
                    row_payload[legacy_key] = row_cells_payload[col_index]["text"]
            row_payload["confidence"] = {
                header_lookup[idx]["key"]: row_cells_payload[idx]["confidence"]
                for idx in range(len(row_cells_payload))
            }
            data_rows.append(row_payload)
        
        column_header_map = {f"column_{idx + 1}": header["label"] for idx, header in enumerate(headers)}
        return {
            "shg_mbk_id": shg_mbk_id,
            "total_rows": len(data_rows),
            "total_columns": len(headers),
            "column_headers": headers,
            "header_rows": header_rows,
            "column_headers_legacy": column_header_map,
            "data_rows": data_rows
        }

    def _structure_with_legacy_layout(self, cells: List[Dict], shg_mbk_id: str) -> Dict:
        column_names = {
            0: "member_id",
            1: "member_name",
            2: "this_month_savings",
            3: "total_savings",
        }
        data_cells = cells[1:] if len(cells) > 1 else []
        rows = []
        cells_per_row = len(column_names)
        for i in range(0, len(data_cells), cells_per_row):
            row_cells = data_cells[i:i + cells_per_row]
            if len(row_cells) != cells_per_row:
                break
            row_payload = {
                "row_number": len(rows) + 1,
                "confidence": {}
            }
            for idx, key_name in column_names.items():
                cell = row_cells[idx]
                row_payload[key_name] = cell.get('text', '')
                row_payload["confidence"][key_name] = float(cell.get('confidence', 0.0))
            rows.append(row_payload)
        return {
            "shg_mbk_id": shg_mbk_id,
            "total_rows": len(rows),
            "total_columns": cells_per_row,
            "column_headers": [
                {"index": idx, "key": key_name, "label": label}
                for idx, (key_name, label) in enumerate([
                    ("member_id", "సభ్యురాలి MBK ID"),
                    ("member_name", "సభ్యురాలు పేరు"),
                    ("this_month_savings", "ఈ నెల పొదుపు"),
                    ("total_savings", "ఈ నెల వరకు పొదుపు"),
                ])
            ],
            "header_rows": [[
                {"label": "సభ్యురాలి MBK ID", "col_span": 1, "row_span": 2},
                {"label": "సభ్యురాలు పేరు", "col_span": 1, "row_span": 2},
                {"label": "ఈ నెల పొదుపు", "col_span": 1, "row_span": 2},
                {"label": "ఈ నెల వరకు పొదుపు", "col_span": 1, "row_span": 2},
            ]],
            "column_headers_legacy": {
                "column_1": "సభ్యురాలి MBK ID",
                "column_2": "సభ్యురాలు పేరు",
                "column_3": "ఈ నెల పొదుపు",
                "column_4": "ఈ నెల వరకు పొదుపు"
            },
            "data_rows": rows
        }

    def log_table_output(self, structured_data: Dict, context: str = "") -> None:
        if not structured_data:
            return
        raw_headers = structured_data.get("column_headers")
        legacy_headers = structured_data.get("column_headers_legacy")
        header_labels: List[str] = []
        if isinstance(raw_headers, list) and raw_headers:
            sorted_headers = sorted(raw_headers, key=lambda h: h.get("index", 0))
            header_labels = [h.get("label") or h.get("key") or f"Column {idx + 1}"
                             for idx, h in enumerate(sorted_headers)]
        elif isinstance(raw_headers, dict) and raw_headers:
            header_labels = [label for _, label in sorted(raw_headers.items(), key=lambda item: item[0])]
        elif isinstance(legacy_headers, dict):
            header_labels = [label for _, label in sorted(legacy_headers.items(), key=lambda item: item[0])]
        else:
            header_labels = ["Column 1", "Column 2", "Column 3", "Column 4"]
        data_rows = structured_data.get("data_rows", [])
        def row_to_values(row: Dict) -> List[str]:
            if isinstance(row.get("cells"), list):
                ordered_cells = sorted(row["cells"], key=lambda c: c.get("col_index", 0))
                return [cell.get("text", "") for cell in ordered_cells]
            return [row.get(label, "") for label in header_labels]
        row_values = [row_to_values(row) for row in data_rows]
        col_widths = []
        for idx, header_text in enumerate(header_labels):
            max_cell = max(
                [len(str(values[idx])) for values in row_values if idx < len(values)] + [len(header_text)],
                default=len(header_text)
            )
            col_widths.append(max_cell + 2)
        def format_row(values: List[str]) -> str:
            padded = []
            for idx, value in enumerate(values):
                width = col_widths[idx] if idx < len(col_widths) else 10
                padded.append(f" {str(value).strip():<{width}} ")
            return "|" + "|".join(padded) + "|"
        horizontal_rule = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        logger.info("\n📋 Scanned Table Output %s", f"({context})" if context else "")
        logger.info(horizontal_rule)
        logger.info(format_row(header_labels))
        logger.info(horizontal_rule)
        for row in data_rows:
            logger.info(format_row(row_to_values(row)))
        logger.info(horizontal_rule)

    def process_cells_with_google_vision(self, cells: List[Dict]) -> List[Dict]:
        """Process cells with optimized batch OCR using TrOCR"""
        logger.info(f"Running TrOCR on {len(cells)} cells...")
        if not self.ocr_client.is_configured():
            logger.error("TrOCR model is not configured — OCR results will be empty.")
            return self._process_without_ocr(cells)
        processed_cells = []
        try:
            cell_images = []
            valid_cells = []
            for cell in cells:
                cell_image = cell.get('image')
                if cell_image is not None:
                    cell_images.append(cell_image)
                    valid_cells.append(cell)
                else:
                    processed_cells.append({
                        'debug_id': int(cell.get('debug_id', -1)),
                        'text': '',
                        'confidence': 0.0,
                        'row': int(cell.get('row', -1)),
                        'col': int(cell.get('col', -1)),
                        'x': int(cell.get('x', 0)),
                        'y': int(cell.get('y', 0)),
                        'bbox': cell.get('bbox')
                    })
            if cell_images:
                # Use optimized batch recognition with GPU processing
                predictions = self.ocr_client.batch_recognize(cell_images)
                for idx, (cell, (text, confidence)) in enumerate(zip(valid_cells, predictions)):
                    processed_cells.append({
                        'debug_id': int(cell.get('debug_id', -1)),
                        'text': str(text),
                        'confidence': float(confidence),
                        'row': int(cell.get('row', -1)),
                        'col': int(cell.get('col', -1)),
                        'x': int(cell.get('x', 0)),
                        'y': int(cell.get('y', 0)),
                        'bbox': cell.get('bbox')
                    })
        except Exception as e:
            logger.error(f"Error during TrOCR processing: {e}")
            logger.error(traceback.format_exc())
            return self._process_without_ocr(cells)
        logger.info(f"✓ Completed TrOCR on {len(processed_cells)} cells")
        return processed_cells
    
    async def process_cells_with_google_vision_async(self, cells: List[Dict]) -> List[Dict]:
        """Async version for even better performance with TrOCR"""
        logger.info(f"Running async TrOCR on {len(cells)} cells...")
        if not self.ocr_client.is_configured():
            logger.error("TrOCR model is not configured — OCR results will be empty.")
            return self._process_without_ocr(cells)
        processed_cells = []
        try:
            cell_images = []
            valid_cells = []
            for cell in cells:
                cell_image = cell.get('image')
                if cell_image is not None:
                    cell_images.append(cell_image)
                    valid_cells.append(cell)
                else:
                    processed_cells.append({
                        'debug_id': int(cell.get('debug_id', -1)),
                        'text': '',
                        'confidence': 0.0,
                        'row': int(cell.get('row', -1)),
                        'col': int(cell.get('col', -1)),
                        'x': int(cell.get('x', 0)),
                        'y': int(cell.get('y', 0)),
                        'bbox': cell.get('bbox')
                    })
            if cell_images:
                # Use async batch recognition for maximum performance with GPU
                predictions = await self.ocr_client.batch_recognize_async(cell_images)
                for idx, (cell, (text, confidence)) in enumerate(zip(valid_cells, predictions)):
                    processed_cells.append({
                        'debug_id': int(cell.get('debug_id', -1)),
                        'text': str(text),
                        'confidence': float(confidence),
                        'row': int(cell.get('row', -1)),
                        'col': int(cell.get('col', -1)),
                        'x': int(cell.get('x', 0)),
                        'y': int(cell.get('y', 0)),
                        'bbox': cell.get('bbox')
                    })
        except Exception as e:
            logger.error(f"Error during async TrOCR processing: {e}")
            logger.error(traceback.format_exc())
            return self._process_without_ocr(cells)
        logger.info(f"✓ Completed async TrOCR on {len(processed_cells)} cells")
        return processed_cells

    def _process_without_ocr(self, cells: List[Dict]) -> List[Dict]:
        processed_cells = []
        for cell in cells:
            processed_cells.append({
                'debug_id': int(cell.get('debug_id', -1)),
                'text': '',
                'confidence': 0.0,
                'row': int(cell.get('row', -1)),
                'col': int(cell.get('col', -1)),
                'x': int(cell.get('x', 0)),
                'y': int(cell.get('y', 0)),
                'bbox': cell.get('bbox'),
                'error': 'Model not loaded'
            })
        return processed_cells

# Endpoints

@app.get('/OCR/api/health')
async def health_check(request: Request):
    try:
        processor = request.app.state.ocr_processor
        ocr_ready = processor.ocr_client.is_configured()
        device_info = "unknown"
        if torch and torch.cuda.is_available():
            device_info = f"cuda:{torch.cuda.get_device_name(0)}"
        elif torch:
            device_info = "cpu"
    except Exception:
        ocr_ready = False
        device_info = "unknown"
    
    status = {
        "status": "healthy",
        "pdf_support": PDF_SUPPORT,
        "validation_enabled": True,
        "trocr_configured": ocr_ready,
        "device": device_info
    }
    logger.info(f"Health check: {status}")
    return status

async def process_single_page_async(processor: OCRProcessor, image_path: Path, filename: str, page_idx: int, total_pages: int) -> Dict:
    """Process a single page asynchronously"""
    try:
        logger.info(f"\n  [Page {page_idx + 1}/{total_pages}] Processing: {image_path.name}")
        # Run image processing in thread pool (CPU-bound)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            processor.process_single_image,
            str(image_path)
        )
        
        if result['success']:
            cells = result.get('cells', [])
            # Use async OCR processing for better performance
            processed_cells = await processor.process_cells_with_google_vision_async(cells)
            structured_data = processor.structure_shg_table_data(processed_cells)
            processor.log_table_output(structured_data, context=f"File: {filename}, Page: {page_idx + 1}")
            
            return {
                'success': True,
                'page': int(page_idx + 1),
                'total_pages': int(total_pages),
                'validation': result.get('validation'),
                'cells': processed_cells,
                'table_data': structured_data,
                'total_cells': int(len(processed_cells))
            }
        else:
            return {
                'success': False,
                'page': int(page_idx + 1),
                'total_pages': int(total_pages),
                'error': result.get('error'),
                'validation': result.get('validation')
            }
    except Exception as e:
        logger.error(f"Error processing page {page_idx + 1}: {e}")
        return {
            'success': False,
            'page': int(page_idx + 1),
            'total_pages': int(total_pages),
            'error': str(e)
        }

async def process_single_file_async(processor: OCRProcessor, file_processor: FileProcessor, 
                                   uploaded_file: UploadFile, file_idx: int, total_files: int,
                                   temp_dirs: List[Path], temp_files: List[Path]) -> Dict:
    """Process a single file asynchronously"""
    if not uploaded_file.filename:
        return None
            
    try:
        logger.info(f"\n[File {file_idx + 1}/{total_files}] Processing: {uploaded_file.filename}")
            
        file_id = uuid.uuid4().hex
        temp_dir = TEMP_FOLDER / file_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dirs.append(temp_dir)
        
        upload_path = temp_dir / uploaded_file.filename
        
        # Save file asynchronously
        def save_file():
            with open(upload_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, save_file)
        temp_files.append(upload_path)
        
        logger.info(f"  Saved to: {upload_path}")
        
        images_to_process = []
        
        if file_processor.is_pdf(uploaded_file.filename):
            if not PDF_SUPPORT:
                return {
                    'success': False,
                    'filename': uploaded_file.filename,
                    'error': 'PDF support not available'
                }
            logger.info("  File type: PDF")
            try:
                images_to_process = await loop.run_in_executor(
                    None,
                    file_processor.pdf_to_images,
                    str(upload_path),
                    temp_dir
                )
                logger.info(f"  Converted to {len(images_to_process)} images")
            except Exception as e:
                return {
                    'success': False,
                    'filename': uploaded_file.filename,
                    'error': f'PDF conversion failed: {str(e)}'
                }
        elif file_processor.is_image(uploaded_file.filename):
            logger.info("  File type: Image")
            converted = await loop.run_in_executor(
                None,
                file_processor.validate_and_convert_image,
                str(upload_path),
                temp_dir
            )
            if converted:
                images_to_process = [converted]
            else:
                return {
                    'success': False,
                    'filename': uploaded_file.filename,
                    'error': 'Invalid image format'
                }
        else:
            return {
                'success': False,
                'filename': uploaded_file.filename,
                'error': 'Unsupported file format'
            }
        
        # Process all pages in parallel
        page_tasks = [
            process_single_page_async(
                processor, image_path, uploaded_file.filename, 
                page_idx, len(images_to_process)
            )
            for page_idx, image_path in enumerate(images_to_process)
        ]
        file_results = await asyncio.gather(*page_tasks)
        
        return {
            'filename': uploaded_file.filename,
            'file_type': 'pdf' if file_processor.is_pdf(uploaded_file.filename) else 'image',
            'total_pages': int(len(images_to_process)),
            'pages': file_results,
            'success': all(r['success'] for r in file_results)
        }
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.filename}: {e}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'filename': uploaded_file.filename,
            'error': str(e)
        }

@app.post('/OCR/api/extract-tables')
async def extract_tables(request: Request, file: List[UploadFile] = File(...)):
    temp_files = []
    temp_dirs = []
    
    try:
        logger.info("="*80)
        logger.info("Received extract-tables request")
        
        if not file:
            return JSONResponse(content={"success": False, "error": "No files uploaded"}, status_code=400)
        
        logger.info(f"Processing {len(file)} uploaded file(s) - OPTIMIZED PARALLEL MODE")
        
        # Use shared processor instance
        processor = request.app.state.ocr_processor
        file_processor = FileProcessor()
        
        # Process all files in parallel
        file_tasks = [
            process_single_file_async(
                processor, file_processor, uploaded_file, 
                file_idx + 1, len(file), temp_dirs, temp_files
            )
            for file_idx, uploaded_file in enumerate(file)
            if uploaded_file.filename
        ]
        
        all_results = await asyncio.gather(*file_tasks)
        all_results = [r for r in all_results if r is not None]
        
        total_pages_processed = sum(len(r.get('pages', [])) for r in all_results)
        successful_pages = sum(
            sum(1 for p in r.get('pages', []) if p.get('success'))
            for r in all_results
        )
        
        response = {
            'success': True,
            'total_files': int(len(file)),
            'total_pages_processed': int(total_pages_processed),
            'successful_pages': int(successful_pages),
            'files': all_results
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing complete: {successful_pages}/{total_pages_processed} pages successful")
        logger.info(f"{'='*80}\n")
        
        return make_json_serializable(response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)
        
    finally:
        # Cleanup temp files with individual error handling
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        for temp_dir in temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")

@app.post('/OCR/api/validate-only')
async def validate_only(file: UploadFile = File(...)):
    temp_files = []
    try:
        if not file.filename:
            return JSONResponse(content={"success": False, "error": "No file selected"}, status_code=400)
        
        # Validate file upload
        file_content = await file.read()
        is_valid, error_msg = validate_file_upload(file.filename, len(file_content))
        if not is_valid:
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        
        # Use sanitized filename
        sanitized_name = sanitize_filename(file.filename)
        temp_path = TEMP_FOLDER / f"{uuid.uuid4().hex}_{sanitized_name}"
        
        # Write file asynchronously
        async with asyncio.Lock():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: temp_path.write_bytes(file_content)
            )
        temp_files.append(temp_path)
        
        # Run validation in executor (CPU-bound)
        validator = SHGImageValidator(debug=False)
        loop = asyncio.get_running_loop()
        validation_result = await loop.run_in_executor(
            None,
            validator.validate_image,
            str(temp_path)
        )
        
        return {
            'success': True,
            'validation': make_json_serializable(validation_result)
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)
        
    finally:
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")

@app.post('/OCR/api/financial/upload')
async def upload_financial_file(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return JSONResponse(content={"success": False, "error": "No file selected"}, status_code=400)
        
        # Validate file type
        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            return JSONResponse(content={"success": False, "error": "Invalid file type. Please upload an Excel file."}, status_code=400)
        
        # Read and validate file
        file_content = await file.read()
        is_valid, error_msg = validate_file_upload(file.filename, len(file_content))
        if not is_valid:
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)

        # Write file asynchronously
        file_path = FINANCIAL_FOLDER / "financial_data.xlsx"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: file_path.write_bytes(file_content)
        )

        logger.info(f"Financial analytics file uploaded: {file_path}")
        return {
            "success": True,
            "message": "Financial data uploaded successfully",
            "path": str(file_path)
        }
    except Exception as e:
        logger.error(f"Financial upload error: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get('/OCR/api/financial/data')
async def get_financial_data(
    district: Optional[str] = Query(None),
    mandal: Optional[str] = Query(None),
    village: Optional[str] = Query(None),
    year: Optional[str] = Query(None),
    month: Optional[str] = Query(None)
):
    try:
        file_path = FINANCIAL_FOLDER / "financial_data.xlsx"
        if not file_path.exists():
            return JSONResponse(content={"success": False, "error": "No financial data found.", "data": None}, status_code=404)

        # Read Excel file in executor (I/O-bound)
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, pd.read_excel, file_path)
        df.columns = df.columns.str.strip()

        # Apply all filters in a single pass for better performance
        filter_conditions = []
        if district and 'District' in df.columns:
            filter_conditions.append(df['District'].astype(str).str.lower() == district.lower())
        if mandal and 'Mandal' in df.columns:
            filter_conditions.append(df['Mandal'].astype(str).str.lower() == mandal.lower())
        if village and 'Village' in df.columns:
            filter_conditions.append(df['Village'].astype(str).str.lower() == village.lower())
        if year and 'Year' in df.columns:
            filter_conditions.append(df['Year'].astype(str) == str(year))
        if month and 'Month' in df.columns:
            filter_conditions.append(df['Month'].astype(str).str.lower() == str(month).lower())
        
        # Combine all filters
        if filter_conditions:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter &= condition
            filtered_df = df[combined_filter]
        else:
            filtered_df = df.copy()

        if filtered_df.empty:
            return {
                "success": True,
                "data": {
                    "savings": {"this_month": 0, "total": 0},
                    "loan_portfolio": [],
                    "loan_type_distribution": [],
                    "repayment_trends": [],
                    "district_shg_loans": [],
                    "district_savings": [],
                    "district_new_loans": [],
                    "district_summaries": {}
                },
                "message": "No data matches filters"
            }

        # ... (Include helper functions column_sum, flexible_column_sum, etc. from app.py)
        # For brevity, I'm simplifying the logic here by assuming the helper functions are defined or inlined.
        # Since I cannot see the full helper functions in the previous turn, I will assume I need to copy them.
        # I will paste the helper functions here.
        
        def column_sum(frame, column_name):
            if column_name not in frame.columns:
                return 0.0
            return float(pd.to_numeric(frame[column_name], errors='coerce').fillna(0).sum())

        def flexible_column_sum(frame, candidates):
            if frame is None or frame.empty:
                return 0.0
            normalized_cols = {str(col).strip().lower(): col for col in frame.columns}
            total = 0.0
            for candidate in candidates:
                key = str(candidate).strip().lower()
                actual = normalized_cols.get(key)
                if not actual:
                    continue
                value = column_sum(frame, actual)
                if value != 0:
                    total = value
                    break
            return total

        COLUMN_SYNONYMS = {
            "Total Savings Balance": ["Total Savings Balance"],
            "This Month Savings": ["This Month Savings"],
            "Total - SHG Loan Balance": ["Total - SHG Loan Balance"],
            "Total - Bank Loan Balance": ["Total - Bank Loan Balance"],
            "Total - Streenidhi Micro Loan Balance": ["Total - Streenidhi Micro Loan Balance", "Total Streenidhi Micro Loan Balance"],
            "Total - Streenidhi Tenny Loan Balance": ["Total - Streenidhi Tenny Loan Balance", "Total Streenidhi Tenny Loan Balance"],
            "Total - Unnathi SCSP Loan Balance": ["Total - Unnathi SCSP Loan Balance", "Total Unnathi SCSP Loan Balance"],
            "Total - Unnathi TSP Loan Balance": ["Total - Unnathi TSP Loan Balance", "Total Unnathi TSP Loan Balance"],
            "Total - CIF Loan Balance": ["Total - CIF Loan Balance", "Total CIF Loan Balance"],
            "Total - VO Loan Balance": ["Total - VO Loan Balance"],
            "This Month SHG Paid Loan": ["This Month SHG Paid Loan"],
            "This Month Bank Loan Paid": ["This Month Bank Loan Paid"],
            "This Month Streenidhi Micro Loan Paid": ["This Month Streenidhi Micro Loan Paid"],
            "This Month Streenidhi Tenny Loan Paid": ["This Month Streenidhi Tenny Loan Paid"],
            "This Month Unnathi SCSP Loan Paid": ["This Month Unnathi SCSP Loan Paid"],
            "This Month Unnathi TSP Loan Paid": ["This Month Unnathi TSP Loan Paid"],
            "This Month CIF Loan Paid": ["This Month CIF Loan Paid"],
            "This Month VO Loan Paid": ["This Month VO Loan Paid"],
            "New Total": ["New Total"],
            "Penalties": ["Penalties", "Penalty", "This Month Penalties"],
            "Entry Membership Fee": ["Entry Membership Fee", "Membership Entry Fee"],
            "Savings Returned": ["Savings Returned", "This Month Savings Returned"],
        }

        def column_sum_with_synonyms(frame, canonical_key):
            candidates = COLUMN_SYNONYMS.get(canonical_key, [canonical_key])
            return flexible_column_sum(frame, candidates)

        savings = {
            "this_month": column_sum_with_synonyms(filtered_df, "This Month Savings"),
            "total": column_sum_with_synonyms(filtered_df, "Total Savings Balance"),
        }

        loan_types = [
            ('SHG Loans', 'Total - SHG Loan Balance'),
            ('Bank Loans', 'Total - Bank Loan Balance'),
            ('Streenidhi Micro', 'Total - Streenidhi Micro Loan Balance'),
            ('Streenidhi Tenny', 'Total - Streenidhi Tenny Loan Balance'),
            ('Unnathi SCSP', 'Total - Unnathi SCSP Loan Balance'),
            ('Unnathi TSP', 'Total - Unnathi TSP Loan Balance'),
            ('CIF Loans', 'Total - CIF Loan Balance'),
            ('VO Loans', 'Total - VO Loan Balance')
        ]

        loan_portfolio = []
        for name, canonical_key in loan_types:
            value = column_sum_with_synonyms(filtered_df, canonical_key)
            if value > 0:
                loan_portfolio.append({"name": name, "value": value})

        loan_type_distribution = []
        possible_names = ['new loan type', 'newloantype', 'loan type', 'loantype', 'new loan', 'newloan']
        new_loan_type_col = 'New Loan Type' if 'New Loan Type' in filtered_df.columns else None
        if new_loan_type_col is None:
            for col in filtered_df.columns:
                col_lower = col.strip().lower()
                if any(name in col_lower for name in possible_names):
                    new_loan_type_col = col
                    break
        
        if new_loan_type_col:
            try:
                type_counts = filtered_df[new_loan_type_col].fillna("").astype(str).str.strip().replace("", np.nan).dropna().value_counts()
                loan_type_distribution = [{"name": str(name), "value": int(count)} for name, count in type_counts.items() if str(name).strip()]
                loan_type_distribution.sort(key=lambda x: x['value'], reverse=True)
            except Exception:
                loan_type_distribution = []

        repayment_defs = [
            ("This Month Savings", "This Month Savings"),
            ("This Month SHG Paid Loan", "This Month SHG Paid Loan"),
            ("This Month Bank Loan Paid", "This Month Bank Loan Paid"),
            ("This Month Streenidhi Micro Loan Paid", "This Month Streenidhi Micro Loan Paid"),
            ("This Month Streenidhi Tenny Loan Paid", "This Month Streenidhi Tenny Loan Paid"),
            ("This Month Unnathi SCSP Loan Paid", "This Month Unnathi SCSP Loan Paid"),
            ("This Month Unnathi TSP Loan Paid", "This Month Unnathi TSP Loan Paid"),
            ("This Month CIF Loan Paid", "This Month CIF Loan Paid"),
            ("This Month VO Loan Paid", "This Month VO Loan Paid"),
            ("Penalty (Current Month)", "Penalties"),
            ("Membership Entry Fee (Current Month)", "Entry Membership Fee"),
            ("Savings Returned (Current Month)", "Savings Returned"),
        ]
        repayment_trends = [{"name": d, "paid": column_sum_with_synonyms(filtered_df, k)} for d, k in repayment_defs]

        district_shg_loans = []
        district_savings = []
        district_new_loans = []
        district_summaries = {}
        
        CUSTOM_TOTAL_COLUMNS = [
            "Total Savings Balance", "Total - SHG Loan Balance", "Total - Bank Loan Balance",
            "Total - Streenidhi Micro Loan Balance", "Total - Streenidhi Tenny Loan Balance",
            "Total - Unnathi SCSP Loan Balance", "Total - Unnathi TSP Loan Balance",
            "Total - CIF Loan Balance", "Total - VO Loan Balance", "New Total",
            "This Month Savings", "This Month SHG Paid Loan", "This Month Bank Loan Paid",
            "This Month Streenidhi Micro Loan Paid", "This Month Streenidhi Tenny Loan Paid",
            "This Month Unnathi SCSP Loan Paid", "This Month Unnathi TSP Loan Paid",
            "This Month CIF Loan Paid", "This Month VO Loan Paid", "Penalties",
            "Entry Membership Fee", "Savings Returned"
        ]

        if 'District' in filtered_df.columns:
            all_districts = filtered_df['District'].dropna().astype(str).str.strip().unique()
            for district_name in all_districts:
                if not district_name or district_name.lower() == 'nan': continue
                group = filtered_df[filtered_df['District'].astype(str).str.strip() == district_name]
                
                shg_balance = column_sum(group, 'Total - SHG Loan Balance')
                if shg_balance > 0:
                    district_shg_loans.append({"name": district_name, "value": shg_balance})
                
                savings_balance = column_sum(group, 'Total Savings Balance')
                if savings_balance > 0:
                    district_savings.append({"name": district_name, "value": savings_balance})
                
                new_loan_count = 0
                if new_loan_type_col and new_loan_type_col in group.columns:
                    new_loan_count = group[new_loan_type_col].astype(str).str.strip().replace('', np.nan).replace('nan', np.nan).notna().sum()
                elif 'New Loan Type' in group.columns:
                    new_loan_count = group['New Loan Type'].astype(str).str.strip().replace('', np.nan).replace('nan', np.nan).notna().sum()
                
                district_new_loans.append({"name": district_name, "value": int(new_loan_count)})
                
                mandal_count = group["Mandal"].dropna().nunique() if "Mandal" in group.columns else 0
                village_count = group["Village"].dropna().nunique() if "Village" in group.columns else 0
                
                column_totals = {k: column_sum_with_synonyms(group, k) for k in CUSTOM_TOTAL_COLUMNS}
                
                district_summaries[district_name] = {
                    "district": district_name,
                    "forms": int(len(group)),
                    "mandals": mandal_count,
                    "villages": village_count,
                    "savings_total": float(savings_balance),
                    "column_totals": column_totals
                }
        
        return {
            "success": True,
            "data": {
                "savings": savings,
                "loan_portfolio": loan_portfolio,
                "loan_type_distribution": loan_type_distribution,
                "repayment_trends": repayment_trends,
                "district_shg_loans": sorted(district_shg_loans, key=lambda x: x['name']),
                "district_savings": sorted(district_savings, key=lambda x: x['name']),
                "district_new_loans": sorted(district_new_loans, key=lambda x: x['name']),
                "district_summaries": district_summaries
            }
        }
    except Exception as e:
        logger.error(f"Financial data fetch error: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.post('/OCR/api/analytics/upload')
async def upload_analytics_file(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return JSONResponse(content={"success": False, "error": "No file selected"}, status_code=400)
        
        # Validate file type
        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            return JSONResponse(content={"success": False, "error": "Invalid file type."}, status_code=400)
        
        # Read and validate file
        file_content = await file.read()
        is_valid, error_msg = validate_file_upload(file.filename, len(file_content))
        if not is_valid:
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        
        # Write file asynchronously
        file_path = ANALYTICS_FOLDER / "analytics_data.xlsx"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: file_path.write_bytes(file_content)
        )
            
        return {"success": True, "message": "Analytics data uploaded successfully", "path": str(file_path)}
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.post('/OCR/api/analytics/load-sample')
def load_sample_analytics():
    try:
        sample_file = ANALYTICS_FOLDER / "Sample Data SHG Data Capture.xlsx"
        target_file = ANALYTICS_FOLDER / "analytics_data.xlsx"
        if sample_file.exists():
            shutil.copy2(str(sample_file), str(target_file))
            return {"success": True, "message": "Sample analytics data loaded successfully", "path": str(target_file)}
        else:
            return JSONResponse(content={"success": False, "error": "Sample file not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.post('/OCR/api/analytics/update')
async def update_analytics_data(request: Request):
    try:
        data = await request.json()
        # Placeholder for actual update logic
        logger.info(f"Received analytics update: {len(data)} items")
        return {"success": True, "message": "Analytics updated successfully"}
    except Exception as e:
        logger.error(f"Error updating analytics: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get('/OCR/api/analytics/data')
async def get_analytics_data(
    district: Optional[str] = Query(None),
    mandal: Optional[str] = Query(None),
    village: Optional[str] = Query(None),
    year: Optional[str] = Query(None),
    month: Optional[str] = Query(None)
):
    """Serve Data Capture Analytics with Summary Information"""
    try:
        # Check for sample file first (as specified by user), then try analytics_data.xlsx
        sample_file = ANALYTICS_FOLDER / "Sample Data SHG Data Capture.xlsx"
        file_path = ANALYTICS_FOLDER / "analytics_data.xlsx"
        
        logger.info(f"Checking for sample file at: {sample_file}")
        logger.info(f"Sample file exists: {sample_file.exists()}")
        
        # Prioritize sample file if it exists
        if sample_file.exists():
            logger.info(f"Using sample file: {sample_file}")
            file_path = sample_file
        elif file_path.exists():
            logger.info(f"Using analytics_data.xlsx: {file_path}")
        else:
            logger.error(f"Neither sample file nor analytics_data.xlsx found in {ANALYTICS_FOLDER}")
            return JSONResponse(content={
                "success": False,
                "error": "No analytics data found. Please ensure Excel file is in analytics_data folder.",
                "data": None
            }, status_code=404)

        # Read Excel file in executor (I/O-bound)
        loop = asyncio.get_running_loop()
        try:
            df = await loop.run_in_executor(None, pd.read_excel, file_path)
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            logger.error(traceback.format_exc())
            return JSONResponse(content={
                "success": False,
                "error": f"Error reading Excel file: {str(e)}",
                "data": None
            }, status_code=500)
        
        if df.empty:
            return JSONResponse(content={
                "success": False,
                "error": "Excel file is empty",
                "data": None
            }, status_code=400)
        
        df.columns = df.columns.str.strip()
        
        # Log available columns for debugging
        logger.info(f"Analytics Excel columns found: {list(df.columns)}")
        logger.info(f"Total rows: {len(df)}")

        # Helper function to find column by name (case-insensitive, partial match)
        def find_column(possible_names):
            try:
                for name in possible_names:
                    # Exact match (case-insensitive)
                    for col in df.columns:
                        if str(col).strip().lower() == name.lower():
                            return col
                    # Partial match
                    for col in df.columns:
                        if name.lower() in str(col).strip().lower():
                            return col
                return None
            except Exception as e:
                logger.error(f"Error in find_column: {e}")
                return None

        # Find actual column names
        district_col = find_column(['District', 'Districts', 'District Name'])
        mandal_col = find_column(['Mandal', 'Mandals', 'Mandal Name'])
        village_col = find_column(['Village', 'Villages', 'Village Name'])
        year_col = find_column(['Year', 'Years'])
        month_col = find_column(['Month', 'Months', 'Month Name'])

        # Apply all filters in a single pass for better performance
        filter_conditions = []
        try:
            if district and district_col:
                filter_conditions.append(filtered_df[district_col].astype(str).str.lower() == district.lower())
            if mandal and mandal_col:
                filter_conditions.append(filtered_df[mandal_col].astype(str).str.lower() == mandal.lower())
            if village and village_col:
                filter_conditions.append(filtered_df[village_col].astype(str).str.lower() == village.lower())
            if year and year_col:
                filter_conditions.append(filtered_df[year_col].astype(str) == str(year))
            if month and month_col:
                filter_conditions.append(filtered_df[month_col].astype(str).str.lower() == str(month).lower())
            
            # Combine all filters
            if filter_conditions:
                combined_filter = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    combined_filter &= condition
                filtered_df = df[combined_filter]
            else:
                filtered_df = df.copy()
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            logger.error(traceback.format_exc())
            filtered_df = df.copy()  # Continue with unfiltered data if filter fails

        # Find numeric data columns from Excel - using exact column names from the Excel file
        total_forms_col = find_column(['Total Forms', 'Total Form', 'Forms', 'Total', 'Imports', 'Total Imports'])
        validation_successful_col = find_column(['Validation Successful', 'Validated Successful', 'Successful', 'Validated', 'Valid Forms'])
        validation_failed_col = find_column(['Validation Failed', 'Failed', 'Validation Failures', 'Failed Validations'])
        synced_to_mkb_col = find_column(['Forms Synced to MBK', 'Forms Synced to MKB', 'Synced to MBK', 'Synced to MKB', 'MBK Synced', 'MKB Synced'])
        
        # Failure detail columns
        failed_incorrect_form_col = find_column(['Failed Incorrect Form', 'Incorrect Form', 'Wrong Form'])
        failed_incorrect_values_col = find_column(['Failed Incorrect Values', 'Incorrect Values', 'Wrong Values'])
        failed_missing_fields_col = find_column(['Failed Missing Fields', 'Missing Fields', 'Missing Data'])
        failed_image_quality_col = find_column(['Failed Image Quality', 'Image Quality', 'Poor Image Quality'])
        
        # Log found columns for debugging
        logger.info(f"Found columns - Total Forms: {total_forms_col}, Validation Successful: {validation_successful_col}, "
                   f"Validation Failed: {validation_failed_col}, Synced to MBK: {synced_to_mkb_col}")
        logger.info(f"Failure detail columns - Incorrect Form: {failed_incorrect_form_col}, "
                   f"Incorrect Values: {failed_incorrect_values_col}, Missing Fields: {failed_missing_fields_col}, "
                   f"Image Quality: {failed_image_quality_col}")
        
        # Calculate summary statistics by summing numeric columns
        total_imports = 0
        validation_successful_count = 0
        validation_failed_count = 0
        synced_to_mkb_count = 0
        
        try:
            if total_forms_col:
                # Sum the Total Forms column
                total_imports = int(filtered_df[total_forms_col].fillna(0).astype(float).sum())
                logger.info(f"Total imports calculated from '{total_forms_col}': {total_imports}")
            else:
                # Fallback: count rows if Total Forms column not found
                total_imports = len(filtered_df)
                logger.warning("Total Forms column not found, using row count as fallback")
            
            if validation_successful_col:
                validation_successful_count = int(filtered_df[validation_successful_col].fillna(0).astype(float).sum())
                logger.info(f"Validation successful calculated from '{validation_successful_col}': {validation_successful_count}")
            else:
                validation_successful_count = 0
                logger.warning("Validation Successful column not found")
            
            if validation_failed_col:
                validation_failed_count = int(filtered_df[validation_failed_col].fillna(0).astype(float).sum())
                logger.info(f"Validation failed calculated from '{validation_failed_col}': {validation_failed_count}")
            elif validation_successful_col and total_forms_col:
                # Calculate failed as total - successful
                validation_failed_count = max(0, total_imports - validation_successful_count)
                logger.info(f"Validation failed calculated as difference: {validation_failed_count}")
            else:
                validation_failed_count = 0
                logger.warning("Validation Failed column not found and cannot be calculated")
            
            if synced_to_mkb_col:
                synced_to_mkb_count = int(filtered_df[synced_to_mkb_col].fillna(0).astype(float).sum())
                logger.info(f"Synced to MBK calculated from '{synced_to_mkb_col}': {synced_to_mkb_count}")
            elif validation_successful_col:
                # Assume successful validations are synced
                synced_to_mkb_count = validation_successful_count
                logger.info(f"Synced to MBK set to validation successful: {synced_to_mkb_count}")
            else:
                synced_to_mkb_count = max(0, total_imports - validation_failed_count)
                logger.info(f"Synced to MBK calculated as difference: {synced_to_mkb_count}")
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            logger.error(traceback.format_exc())
            # Fallback to row count
            total_imports = len(filtered_df)
            validation_successful_count = 0
            validation_failed_count = 0
            synced_to_mkb_count = 0
        
        # Get validation failed details - records where Validation Failed > 0 (limit to 100 during creation)
        validation_failed_details = []
        try:
            if validation_failed_col:
                # Get records where validation failed count > 0
                failed_records = filtered_df[filtered_df[validation_failed_col].fillna(0).astype(float) > 0].copy()
                
                if len(failed_records) > 0:
                    # Include important columns in details
                    detail_cols = []
                    if district_col:
                        detail_cols.append(district_col)
                    if mandal_col:
                        detail_cols.append(mandal_col)
                    if village_col:
                        detail_cols.append(village_col)
                    if year_col:
                        detail_cols.append(year_col)
                    if month_col:
                        detail_cols.append(month_col)
                    if validation_failed_col:
                        detail_cols.append(validation_failed_col)
                    if failed_incorrect_form_col:
                        detail_cols.append(failed_incorrect_form_col)
                    if failed_incorrect_values_col:
                        detail_cols.append(failed_incorrect_values_col)
                    if failed_missing_fields_col:
                        detail_cols.append(failed_missing_fields_col)
                    if failed_image_quality_col:
                        detail_cols.append(failed_image_quality_col)
                    
                    # Add any other relevant columns (limit to avoid too much data)
                    for col in failed_records.columns:
                        if col not in detail_cols and len(detail_cols) < 15:
                            if any(keyword in str(col).lower() for keyword in ['failed', 'error', 'issue', 'form', 'shg']):
                                detail_cols.append(col)
                    
                    available_cols = [col for col in detail_cols if col in failed_records.columns]
                    if available_cols:
                        # Limit to 100 records BEFORE creating the dict for better memory efficiency
                        limited_records = failed_records.head(100)
                        # Replace NaN values with None (which becomes null in JSON) or empty string
                        details_df = limited_records[available_cols].fillna('')
                        validation_failed_details = details_df.to_dict(orient='records')
                        # Clean up any remaining NaN values in the records
                        for record in validation_failed_details:
                            for key, value in record.items():
                                if pd.isna(value) or (isinstance(value, float) and str(value) == 'nan'):
                                    record[key] = ''
        except Exception as e:
            logger.error(f"Error processing validation failed details: {e}")
            logger.error(traceback.format_exc())
            validation_failed_details = []

        # District-wise aggregations using numeric columns
        district_summaries = {}
        try:
            if district_col:
                unique_districts = filtered_df[district_col].dropna().unique()
                for district_name in unique_districts:
                    try:
                        district_name_str = str(district_name).strip()
                        if not district_name_str or district_name_str.lower() == 'nan':
                            continue
                            
                        district_data = filtered_df[filtered_df[district_col] == district_name]
                        
                        # Calculate district statistics using numeric columns
                        district_imports = 0
                        if total_forms_col:
                            district_imports = int(district_data[total_forms_col].fillna(0).astype(float).sum())
                        else:
                            district_imports = len(district_data)
                        
                        district_validation_failed = 0
                        if validation_failed_col:
                            district_validation_failed = int(district_data[validation_failed_col].fillna(0).astype(float).sum())
                        elif validation_successful_col and total_forms_col:
                            district_successful = int(district_data[validation_successful_col].fillna(0).astype(float).sum())
                            district_validation_failed = max(0, district_imports - district_successful)
                        
                        district_synced = 0
                        if synced_to_mkb_col:
                            district_synced = int(district_data[synced_to_mkb_col].fillna(0).astype(float).sum())
                        elif validation_successful_col:
                            district_synced = int(district_data[validation_successful_col].fillna(0).astype(float).sum())
                        else:
                            district_synced = max(0, district_imports - district_validation_failed)
                        
                        district_summaries[district_name_str] = {
                            "district": district_name_str,
                            "imports": district_imports,
                            "validation_failed": district_validation_failed,
                            "synced_to_mkb": district_synced,
                            "success_rate": round((district_synced / district_imports * 100) if district_imports > 0 else 0, 2)
                        }
                    except Exception as e:
                        logger.error(f"Error processing district {district_name}: {e}")
                        logger.error(traceback.format_exc())
                        continue
        except Exception as e:
            logger.error(f"Error in district aggregations: {e}")
            logger.error(traceback.format_exc())

        # Mandal-wise aggregations (if mandal column exists)
        mandal_summaries = {}
        try:
            if mandal_col and district_col:
                # Group by both district and mandal for better organization
                for district_name in filtered_df[district_col].dropna().unique():
                    district_name_str = str(district_name).strip()
                    if not district_name_str or district_name_str.lower() == 'nan':
                        continue
                    
                    district_data = filtered_df[filtered_df[district_col] == district_name]
                    unique_mandals = district_data[mandal_col].dropna().unique()
                    
                    for mandal_name in unique_mandals:
                        try:
                            mandal_name_str = str(mandal_name).strip()
                            if not mandal_name_str or mandal_name_str.lower() == 'nan':
                                continue
                            
                            mandal_data = district_data[district_data[mandal_col] == mandal_name]
                            
                            mandal_imports = 0
                            if total_forms_col:
                                mandal_imports = int(mandal_data[total_forms_col].fillna(0).astype(float).sum())
                            else:
                                mandal_imports = len(mandal_data)
                            
                            mandal_validation_failed = 0
                            if validation_failed_col:
                                mandal_validation_failed = int(mandal_data[validation_failed_col].fillna(0).astype(float).sum())
                            
                            mandal_synced = 0
                            if synced_to_mkb_col:
                                mandal_synced = int(mandal_data[synced_to_mkb_col].fillna(0).astype(float).sum())
                            elif validation_successful_col:
                                mandal_synced = int(mandal_data[validation_successful_col].fillna(0).astype(float).sum())
                            else:
                                mandal_synced = max(0, mandal_imports - mandal_validation_failed)
                            
                            key = f"{district_name_str} - {mandal_name_str}"
                            mandal_summaries[key] = {
                                "district": district_name_str,
                                "mandal": mandal_name_str,
                                "imports": mandal_imports,
                                "validation_failed": mandal_validation_failed,
                                "synced_to_mkb": mandal_synced,
                                "success_rate": round((mandal_synced / mandal_imports * 100) if mandal_imports > 0 else 0, 2)
                            }
                        except Exception as e:
                            logger.error(f"Error processing mandal {mandal_name}: {e}")
                            continue
        except Exception as e:
            logger.error(f"Error in mandal aggregations: {e}")
            logger.error(traceback.format_exc())
        
        # Helper function to safely convert to int, handling NaN
        def safe_int_sum(series, default=0):
            """Safely sum a pandas series and convert to int, handling NaN values."""
            try:
                if series is None:
                    return default
                if hasattr(series, '__len__') and len(series) == 0:
                    return default
                result = series.fillna(0).astype(float).sum()
                # Handle NaN result - check if result is NaN using multiple methods
                import math
                if pd.isna(result) or (isinstance(result, float) and math.isnan(result)) or (result != result):
                    return default
                return int(result) if not math.isnan(result) else default
            except Exception as e:
                logger.warning(f"Error in safe_int_sum: {e}")
                return default
        
        # Prepare chart data grouped by district and year
        chart_data_by_district = {}
        chart_data_by_year = {}
        
        try:
            if district_col:
                # Group by district
                for district_name in filtered_df[district_col].dropna().unique():
                    district_name_str = str(district_name).strip()
                    if not district_name_str or district_name_str.lower() == 'nan':
                        continue
                    
                    district_data = filtered_df[filtered_df[district_col] == district_name]
                    
                    chart_data_by_district[district_name_str] = {
                        "total_forms": safe_int_sum(district_data[total_forms_col] if total_forms_col and total_forms_col in district_data.columns else None),
                        "validation_successful": safe_int_sum(district_data[validation_successful_col] if validation_successful_col and validation_successful_col in district_data.columns else None),
                        "validation_failed": safe_int_sum(district_data[validation_failed_col] if validation_failed_col and validation_failed_col in district_data.columns else None),
                        "failed_incorrect_form": safe_int_sum(district_data[failed_incorrect_form_col] if failed_incorrect_form_col and failed_incorrect_form_col in district_data.columns else None),
                        "failed_incorrect_values": safe_int_sum(district_data[failed_incorrect_values_col] if failed_incorrect_values_col and failed_incorrect_values_col in district_data.columns else None),
                        "failed_missing_fields": safe_int_sum(district_data[failed_missing_fields_col] if failed_missing_fields_col and failed_missing_fields_col in district_data.columns else None),
                        "failed_image_quality": safe_int_sum(district_data[failed_image_quality_col] if failed_image_quality_col and failed_image_quality_col in district_data.columns else None),
                        "synced_to_mkb": safe_int_sum(district_data[synced_to_mkb_col] if synced_to_mkb_col and synced_to_mkb_col in district_data.columns else None)
                    }
            
            if year_col:
                # Group by year
                for year_val in filtered_df[year_col].dropna().unique():
                    year_str = str(year_val).strip()
                    if not year_str or year_str.lower() == 'nan':
                        continue
                    
                    year_data = filtered_df[filtered_df[year_col] == year_val]
                    
                    chart_data_by_year[year_str] = {
                        "total_forms": safe_int_sum(year_data[total_forms_col] if total_forms_col and total_forms_col in year_data.columns else None),
                        "validation_successful": safe_int_sum(year_data[validation_successful_col] if validation_successful_col and validation_successful_col in year_data.columns else None),
                        "validation_failed": safe_int_sum(year_data[validation_failed_col] if validation_failed_col and validation_failed_col in year_data.columns else None),
                        "failed_incorrect_form": safe_int_sum(year_data[failed_incorrect_form_col] if failed_incorrect_form_col and failed_incorrect_form_col in year_data.columns else None),
                        "failed_incorrect_values": safe_int_sum(year_data[failed_incorrect_values_col] if failed_incorrect_values_col and failed_incorrect_values_col in year_data.columns else None),
                        "failed_missing_fields": safe_int_sum(year_data[failed_missing_fields_col] if failed_missing_fields_col and failed_missing_fields_col in year_data.columns else None),
                        "failed_image_quality": safe_int_sum(year_data[failed_image_quality_col] if failed_image_quality_col and failed_image_quality_col in year_data.columns else None),
                        "synced_to_mkb": safe_int_sum(year_data[synced_to_mkb_col] if synced_to_mkb_col and synced_to_mkb_col in year_data.columns else None)
                    }
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            logger.error(traceback.format_exc())
        
        # Ensure all numeric values are valid (no NaN)
        def clean_numeric(value):
            """Clean numeric values, replacing NaN with 0."""
            if pd.isna(value) or (isinstance(value, float) and str(value) == 'nan'):
                return 0
            try:
                return float(value) if not pd.isna(value) else 0
            except (ValueError, TypeError):
                return 0
        
        # Clean summary values
        clean_summary = {
            "total_imports": int(clean_numeric(total_imports)),
            "validation_successful": int(clean_numeric(validation_successful_count)),
            "validation_failed": int(clean_numeric(validation_failed_count)),
            "synced_to_mkb": int(clean_numeric(synced_to_mkb_count)),
            "success_rate": round((synced_to_mkb_count / total_imports * 100) if total_imports > 0 else 0, 2)
        }
        
        # Clean district summaries
        clean_district_summaries = {}
        for district, data in district_summaries.items():
            clean_district_summaries[district] = {
                "district": str(data.get("district", "")),
                "imports": int(clean_numeric(data.get("imports", 0))),
                "validation_failed": int(clean_numeric(data.get("validation_failed", 0))),
                "synced_to_mkb": int(clean_numeric(data.get("synced_to_mkb", 0))),
                "success_rate": round(clean_numeric(data.get("success_rate", 0)), 2)
            }
        
        return {
            "success": True,
            "data": {
                "summary": clean_summary,
                "validation_failed_details": validation_failed_details,  # Already limited to 100
                "district_summaries": clean_district_summaries,
                "mandal_summaries": mandal_summaries,
                "chart_data": {
                    "by_district": chart_data_by_district,
                    "by_year": chart_data_by_year
                },
                "filters_applied": {
                    "district": district,
                    "mandal": mandal,
                    "village": village,
                    "year": year,
                    "month": month
                }
            }
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5002)
