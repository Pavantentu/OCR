from flask import Flask, request, jsonify, Response
from flask_cors import CORS
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
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image
import requests
import pandas as pd
import copy

try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleAuthRequest
except ImportError:
    service_account = None
    GoogleAuthRequest = None

# Import your validation and detection modules
from validate import process_with_validation, SHGImageValidator
from test import SHGFormDetector

# Configure logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('flask_app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# SHG table layout definitions (matching the original template)
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

# Multi-level header rows to mimic the physical form layout
BASE_SHG_HEADER_ROWS = [
    # First row: Title row spanning all columns
    [
        {"label": "………......................... స్వయం స్హయక స్ంఘ  ................. తేదిన జరిగిన స్మావేశ ఆరిిక లావాదేవీలు వివరములు (అనుభందం - II)", "col_span": 17, "row_span": 1},
    ],
    # Main category headers
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


def build_shg_header_rows(shg_mbk_id: str) -> List[List[Dict[str, str]]]:
    """
    Inject a dynamic SHG MBK ID row between the title and category headers while
    preserving the base multi-level header layout.
    """
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


def build_shg_column_headers(total_columns: int) -> List[Dict[str, str]]:
    """
    Create a list of column header dictionaries describing the SHG layout.
    Always returns at least DEFAULT_SHG_COLUMN_COUNT headers.
    """
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

# ============================================================================
# GOOGLE VISION CONFIGURATION
# ============================================================================
GOOGLE_VISION_API_KEY = {

}


GOOGLE_VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
GOOGLE_VISION_REQUEST_TIMEOUT = 20  # seconds
# ============================================================================

app = Flask(__name__)
CORS(app, 
    resources={
        r"/OCR/*": {
            "origins": [
                "http://localhost:5173",
                "https://pavantentu.github.io"
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "max_age": 3600
        },
        r"/api/*": {
            "origins": ["*"],
            "methods": ["GET", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "max_age": 3600
        }
    }
)

# Configuration
TEMP_FOLDER = Path('temp_processing')
UPLOAD_FOLDER = Path('uploads')
RESULT_FOLDER = Path('result')
FINANCIAL_FOLDER = Path(__file__).resolve().parent / 'financial_data'
ANALYTICS_FOLDER = Path(__file__).resolve().parent / 'analytics_data'
CSV_FILE_PATH = Path(__file__).resolve().parent / 'districts-mandals.csv'

# Create necessary folders
for folder in [TEMP_FOLDER, UPLOAD_FOLDER, RESULT_FOLDER, FINANCIAL_FOLDER, ANALYTICS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOCATION DATA (Districts/Mandals/Villages) - Load from CSV
# ============================================================================
# In-memory cache for districts, mandals, and villages
districts_cache = []
mandals_by_district_cache = {}  # Key: district name, Value: list of mandal names
villages_by_mandal_cache = {}  # Key: "district|mandal", Value: list of village names

def load_location_data():
    """Load and parse districts-mandals.csv file"""
    global districts_cache, mandals_by_district_cache, villages_by_mandal_cache
    
    try:
        if not CSV_FILE_PATH.exists():
            logger.warning(f"CSV file not found at {CSV_FILE_PATH}")
            # Try alternative path (in backend directory)
            alt_path = Path(__file__).resolve().parent / 'districts-mandals.csv'
            if alt_path.exists():
                csv_path = alt_path
            else:
                logger.error("Could not find districts-mandals.csv file")
                return
        else:
            csv_path = CSV_FILE_PATH
        
        logger.info(f"Loading location data from: {csv_path}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Normalize column names (handle case variations)
        df.columns = df.columns.str.strip()
        
        # Find column indices (case-insensitive)
        mandal_col = None
        district_col = None
        village_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'mandal' in col_lower:
                mandal_col = col
            elif 'district' in col_lower:
                district_col = col
            elif 'village' in col_lower:
                village_col = col
        
        if not mandal_col or not district_col:
            logger.error(f"Required columns not found. Available columns: {list(df.columns)}")
            return
        
        # Clear existing cache
        districts_cache = []
        mandals_by_district_cache = {}
        villages_by_mandal_cache = {}
        
        districts_set = set()
        
        # Process each row
        for _, row in df.iterrows():
            mandal = str(row[mandal_col]).strip() if pd.notna(row[mandal_col]) else ""
            district = str(row[district_col]).strip() if pd.notna(row[district_col]) else ""
            village = str(row[village_col]).strip() if village_col and pd.notna(row[village_col]) else ""
            
            if not district or not mandal:
                continue
            
            # Add district to set
            districts_set.add(district)
            
            # Add mandal to district map
            if district not in mandals_by_district_cache:
                mandals_by_district_cache[district] = []
            
            if mandal not in mandals_by_district_cache[district]:
                mandals_by_district_cache[district].append(mandal)
            
            # Add village if present
            if village:
                key = f"{district}|{mandal}"
                if key not in villages_by_mandal_cache:
                    villages_by_mandal_cache[key] = []
                
                if village not in villages_by_mandal_cache[key]:
                    villages_by_mandal_cache[key].append(village)
        
        # Convert set to sorted list
        districts_cache = sorted(list(districts_set))
        
        # Sort mandals and villages
        for district in mandals_by_district_cache:
            mandals_by_district_cache[district].sort()
        
        for key in villages_by_mandal_cache:
            villages_by_mandal_cache[key].sort()
        
        logger.info(f"✓ Loaded {len(districts_cache)} districts")
        logger.info(f"✓ Loaded mandals for {len(mandals_by_district_cache)} districts")
        logger.info(f"✓ Loaded villages for {len(villages_by_mandal_cache)} mandals")
        
    except Exception as e:
        logger.error(f"Error loading location data: {e}")
        logger.error(traceback.format_exc())
        # Don't crash the server - just log the error and continue with empty cache
        districts_cache = []
        mandals_by_district_cache = {}
        villages_by_mandal_cache = {}

# Load location data on startup
load_location_data()

# PDF Processing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
    logger.info("PDF support enabled (PyMuPDF)")
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PDF support disabled - install PyMuPDF: pip install PyMuPDF")


def make_json_serializable(obj):
    """
    Convert numpy types and other non-serializable objects to JSON-safe types.
    """
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
    """Handles file processing for images and PDFs"""
    
    @staticmethod
    def is_pdf(filename: str) -> bool:
        """Check if file is a PDF"""
        return filename.lower().endswith('.pdf')
    
    @staticmethod
    def is_image(filename: str) -> bool:
        """Check if file is an image"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return Path(filename).suffix.lower() in valid_extensions
    
    @staticmethod
    def pdf_to_images(pdf_path: str, output_folder: Path) -> List[Path]:
        """
        Convert PDF pages to images
        
        Returns:
            List of image file paths
        """
        if not PDF_SUPPORT:
            raise Exception("PDF support not available - install PyMuPDF")
        
        logger.info(f"Converting PDF to images: {pdf_path}")
        image_paths = []
        
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"PDF has {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Render page to image at high resolution (300 DPI)
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save as temporary image
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
        """
        Validate image format and convert if needed
        
        Returns:
            Path to validated/converted image
        """
        try:
            img = cv2.imread(str(file_path))
            
            if img is None:
                logger.error(f"Could not read image: {file_path}")
                return None
            
            # Convert to standard format (JPEG)
            output_path = output_folder / f"{Path(file_path).stem}_converted.jpg"
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.info(f"Image validated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return None


class GoogleVisionOCR:
    """Lightweight wrapper around Google Vision API for text detection."""

    def __init__(self, credential_payload):
        self.auth_mode = "api_key"
        self.api_key = ""
        self.credentials = None
        self._auth_request = None

        if isinstance(credential_payload, dict):
            if service_account is None or GoogleAuthRequest is None:
                logger.error("google-auth library is required for service account credentials. Install with 'pip install google-auth'.")
            else:
                try:
                    self.credentials = service_account.Credentials.from_service_account_info(
                        credential_payload,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    self._auth_request = GoogleAuthRequest()
                    self.auth_mode = "service_account"
                except Exception as exc:
                    logger.error(f"Failed to load Google service account credentials: {exc}")
        else:
            self.api_key = (credential_payload or "").strip()
            self.auth_mode = "api_key"

        self.endpoint = GOOGLE_VISION_ENDPOINT
        self.session = requests.Session()

    def is_configured(self) -> bool:
        """Return True when credentials (API key or service account) are available."""
        if self.auth_mode == "service_account":
            return self.credentials is not None
        if not self.api_key:
            return False
        return "PASTE_GOOGLE_VISION_API_KEY_HERE" not in self.api_key

    def _encode_image(self, image: np.ndarray) -> Optional[str]:
        """Encode numpy image to base64 for Vision API."""
        if image is None:
            return None
        success, buffer = cv2.imencode('.png', image)
        if not success:
            return None
        return base64.b64encode(buffer).decode('utf-8')

    def recognize_text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Send a single image to Google Vision and return text + pseudo confidence.
        """
        if not self.is_configured():
            logger.error("Google Vision API key is not configured.")
            return "", 0.0

        image_content = self._encode_image(image)
        if not image_content:
            logger.error("Failed to encode image for Google Vision request.")
            return "", 0.0

        payload = {
            "requests": [
                {
                    "image": {"content": image_content},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
                }
            ]
        }

        params = None
        headers = {"Content-Type": "application/json"}

        if self.auth_mode == "service_account":
            if not self.credentials:
                logger.error("Service account credentials not available.")
                return "", 0.0
            try:
                if not self.credentials.valid:
                    if self._auth_request is None:
                        self._auth_request = GoogleAuthRequest()
                    self.credentials.refresh(self._auth_request)
                headers["Authorization"] = f"Bearer {self.credentials.token}"
            except Exception as exc:
                logger.error(f"Failed to refresh Google access token: {exc}")
                return "", 0.0
        else:
            params = {"key": self.api_key}

        try:
            response = self.session.post(
                self.endpoint,
                params=params,
                headers=headers,
                json=payload,
                timeout=GOOGLE_VISION_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error(f"Google Vision request failed: {exc}")
            return "", 0.0

        responses = data.get("responses", [])
        if not responses:
            logger.warning("Google Vision response contained no results.")
            return "", 0.0

        annotation = responses[0]
        text = ""
        confidence = 0.0

        full_text = annotation.get("fullTextAnnotation", {})
        if full_text and full_text.get("text"):
            text = full_text["text"].replace("\n", " ").strip()
            confidence = full_text.get("confidence", 0.9)
        else:
            text_annotations = annotation.get("textAnnotations", [])
            if text_annotations:
                text = text_annotations[0].get("description", "").replace("\n", " ").strip()
                confidence = text_annotations[0].get("score", 0.85)

        return text, float(confidence or 0.0)

    def batch_recognize(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Process multiple images sequentially (Vision API supports batching but per-image keeps payloads small)."""
        results = []
        for idx, image in enumerate(images):
            text, confidence = self.recognize_text(image)
            logger.debug(f"Google Vision OCR [{idx+1}/{len(images)}]: '{text}' (conf={confidence})")
            results.append((text, confidence))
        return results

class OCRProcessor:
    """Handles OCR processing for SHG forms using Google Vision."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize OCR processor.
        
        Args:
            debug: Enable debug logging (does NOT save debug images)
        """
        self.debug = debug
        
        # CRITICAL: Pass debug=False to validator and detector to prevent saving debug images
        self.validator = SHGImageValidator(debug=False)
        
        # Initialize Google Vision client
        self.ocr_client = GoogleVisionOCR(GOOGLE_VISION_API_KEY)
        
        if self.ocr_client.is_configured():
            logger.info("✓ OCR Processor initialized with Google Vision")
        else:
            logger.warning("⚠ Google Vision API key missing — OCR will return empty text until configured")
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        Process a single image through validation pipeline
        
        Returns:
            Dict with validation results and extracted cells
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing image: {image_path}")
        logger.info(f"{'='*70}")
        
        try:
        # Run validation and extraction pipeline without saving debug/training outputs
            result = process_with_validation(
                image_path,
            debug=False,  # No debug files
            training_mode=False,
            return_images=False  # Keep processing in-memory only
            )
            
            if not result or not result.get('success'):
                return {
                    'success': False,
                    'error': result.get('error', 'Validation failed') if result else 'Processing failed',
                    'validation': result.get('validation') if result else None,
                    'image_path': str(image_path)
                }
            
            # Extract cell data
            cells = result.get('cells', [])
            shg_id = result.get('shg_id')
            
            logger.info(f"✓ Successfully processed: {len(cells)} cells extracted")

            logger.info("🔍 DEBUG: Validation result:")
            logger.info(result)
            logger.info("🔍 DEBUG: Extracted cell count: " + str(len(cells)))
            
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
    
    def encode_image_base64(self, image_array: np.ndarray) -> str:
        """Convert numpy array to base64 string"""
        try:
            _, buffer = cv2.imencode('.jpg', image_array)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return ""
    
    def structure_shg_table_data(self, cells: List[Dict]) -> Dict:
        """
        Structure cells into the full SHG form layout (17 columns).
        Falls back to the legacy 4-column view if row/column metadata is missing.
        """
        # Extract SHG_MBK_ID (debug_id=2) and exclude it from table rows
        shg_mbk_id = ""
        for cell in cells:
            if cell.get('debug_id') == 2:
                shg_mbk_id = cell.get('text', '')
                break
        
        rows_by_index: Dict[int, Dict[int, Dict]] = {}
        max_col_index = -1
        has_row_col_metadata = False
        
        for cell in cells:
            # Skip the SHG MBK ID cell (debug_id=2) as it's already extracted separately
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
            
            # Backwards compatibility for clients that still expect named keys
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
        """
        Preserve the previous 4-column behavior for compatibility when row/col
        metadata is not available (older detector builds).
        """
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
        """
        Pretty-print the structured table in the backend terminal along with
        a Telugu reference format so noisy detections can be inspected quickly.
        """
        if not structured_data:
            logger.warning("No structured table data available to log")
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
        if not data_rows:
            logger.warning("Structured table has no rows to display")
        
        def row_to_values(row: Dict) -> List[str]:
            if isinstance(row.get("cells"), list):
                ordered_cells = sorted(row["cells"], key=lambda c: c.get("col_index", 0))
                return [cell.get("text", "") for cell in ordered_cells]
            return [row.get(label, "") for label in header_labels]
        
        row_values = [row_to_values(row) for row in data_rows]
        
        # Determine width per column (header vs cell content)
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
        
        logger.info("📑 Telugu Reference Format (empty cells shown intentionally)")
        logger.info(horizontal_rule)
        logger.info(format_row(header_labels))
        logger.info(horizontal_rule)
        for _ in range(max(1, len(data_rows))):
            logger.info(format_row(["" for _ in header_labels]))
        logger.info(horizontal_rule)
        return
    
    def process_cells_with_google_vision(self, cells: List[Dict]) -> List[Dict]:
        """
        Process cells with Google Vision API - Returns TEXT ONLY (no images in response)
        
        Args:
            cells: List of cell dictionaries with 'image' field
        
        Returns:
            List of cells with 'text' and 'confidence' fields (NO image_base64)
        """
        logger.info(f"Running Google Vision OCR on {len(cells)} cells...")
        
        if not self.ocr_client.is_configured():
            logger.error("Google Vision API key is not configured — OCR results will be empty.")
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
                    logger.warning(f"Cell {cell.get('debug_id')} has no image payload")
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
                    
                    if not text.strip():
                        logger.warning(f"⚠ WARNING: EMPTY OCR result for cell {cell.get('debug_id')}")
                    
                    if (idx + 1) % 20 == 0:
                        logger.info(f"  Progress: {idx + 1}/{len(valid_cells)} cells processed")
        
        except Exception as e:
            logger.error(f"Error during Google Vision OCR processing: {e}")
            logger.error(traceback.format_exc())
            return self._process_without_ocr(cells)
        
        logger.info(f"✓ Completed Google Vision OCR on {len(processed_cells)} cells")
        return processed_cells
    
    def _process_without_ocr(self, cells: List[Dict]) -> List[Dict]:
        """Fallback when OCR is not available."""
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


@app.route('/OCR/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        processor = OCRProcessor(debug=False)
        ocr_ready = processor.ocr_client.is_configured()
    except Exception:
        ocr_ready = False
    
    status = {
        "status": "healthy",
        "pdf_support": PDF_SUPPORT,
        "validation_enabled": True,
        "google_vision_configured": ocr_ready
    }
    logger.info(f"Health check: {status}")
    return jsonify(status)


@app.route('/OCR/api/extract-tables', methods=['POST'])
def extract_tables():
    """
    Main endpoint for table extraction with Google Vision text recognition.
    Returns TEXT ONLY (no images) for each cell.
    """
    temp_files = []
    temp_dirs = []
    
    try:
        logger.info("="*80)
        logger.info("Received extract-tables request")
        
        # Check if files were uploaded
        if 'file' not in request.files and 'files' not in request.files:
            return jsonify({
                "success": False,
                "error": "No files uploaded"
            }), 400
        
        # Get uploaded files
        if 'files' in request.files:
            uploaded_files = request.files.getlist('files')
        else:
            uploaded_files = [request.files['file']]
        
        logger.info(f"Processing {len(uploaded_files)} uploaded file(s)")
        
        # Process each uploaded file
        all_results = []
        
        # Initialize processor with Google Vision OCR
        processor = OCRProcessor(debug=False)
        file_processor = FileProcessor()
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.filename == '':
                continue
            
            logger.info(f"\n[File {file_idx + 1}/{len(uploaded_files)}] Processing: {uploaded_file.filename}")
            
            # Save uploaded file
            file_id = uuid.uuid4().hex
            temp_dir = TEMP_FOLDER / file_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_dirs.append(temp_dir)
            
            upload_path = temp_dir / uploaded_file.filename
            uploaded_file.save(str(upload_path))
            temp_files.append(upload_path)
            
            logger.info(f"  Saved to: {upload_path}")
            
            # Determine file type and convert to images
            images_to_process = []
            
            if file_processor.is_pdf(uploaded_file.filename):
                if not PDF_SUPPORT:
                    all_results.append({
                        'success': False,
                        'filename': uploaded_file.filename,
                        'error': 'PDF support not available'
                    })
                    continue
                
                logger.info("  File type: PDF")
                try:
                    images_to_process = file_processor.pdf_to_images(
                        str(upload_path),
                        temp_dir
                    )
                    logger.info(f"  Converted to {len(images_to_process)} images")
                except Exception as e:
                    all_results.append({
                        'success': False,
                        'filename': uploaded_file.filename,
                        'error': f'PDF conversion failed: {str(e)}'
                    })
                    continue
                    
            elif file_processor.is_image(uploaded_file.filename):
                logger.info("  File type: Image")
                converted = file_processor.validate_and_convert_image(
                    str(upload_path),
                    temp_dir
                )
                if converted:
                    images_to_process = [converted]
                else:
                    all_results.append({
                        'success': False,
                        'filename': uploaded_file.filename,
                        'error': 'Invalid image format'
                    })
                    continue
            else:
                all_results.append({
                    'success': False,
                    'filename': uploaded_file.filename,
                    'error': 'Unsupported file format'
                })
                continue
            
            # Process each image (page)
            file_results = []
            
            for page_idx, image_path in enumerate(images_to_process):
                logger.info(f"\n  [Page {page_idx + 1}/{len(images_to_process)}] Processing: {image_path.name}")
                
                # Run validation and extraction
                result = processor.process_single_image(str(image_path))
                
                if result['success']:
                    # Process cells with Google Vision - Returns TEXT ONLY
                    cells = result.get('cells', [])
                    processed_cells = processor.process_cells_with_google_vision(cells)
                    
                    # Structure data into proper SHG table format
                    structured_data = processor.structure_shg_table_data(processed_cells)
                    processor.log_table_output(
                        structured_data,
                        context=f"File: {uploaded_file.filename}, Page: {page_idx + 1}"
                    )
                    
                    # Build page result with STRUCTURED TABLE DATA
                    page_result = {
                        'success': True,
                        'page': int(page_idx + 1),
                        'total_pages': int(len(images_to_process)),
                        'validation': result.get('validation'),
                        'cells': processed_cells,
                        'table_data': structured_data,  # Structured table with rows/columns
                        'total_cells': int(len(processed_cells))
                    }
                else:
                    page_result = {
                        'success': False,
                        'page': int(page_idx + 1),
                        'total_pages': int(len(images_to_process)),
                        'error': result.get('error'),
                        'validation': result.get('validation')
                    }
                
                file_results.append(page_result)
            
            # Add file-level result
            all_results.append({
                'filename': uploaded_file.filename,
                'file_type': 'pdf' if file_processor.is_pdf(uploaded_file.filename) else 'image',
                'total_pages': int(len(images_to_process)),
                'pages': file_results,
                'success': all(r['success'] for r in file_results)
            })
        
        # Build final response
        total_pages_processed = sum(len(r.get('pages', [])) for r in all_results)
        successful_pages = sum(
            sum(1 for p in r.get('pages', []) if p.get('success'))
            for r in all_results
        )
        
        response = {
            'success': True,
            'total_files': int(len(uploaded_files)),
            'total_pages_processed': int(total_pages_processed),
            'successful_pages': int(successful_pages),
            'files': all_results
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing complete:")
        logger.info(f"  Files: {len(uploaded_files)}")
        logger.info(f"  Pages: {total_pages_processed}")
        logger.info(f"  Success: {successful_pages}/{total_pages_processed}")
        logger.info(f"{'='*80}\n")
        
        # Make response JSON-safe
        response = make_json_serializable(response)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        
    finally:
        # Cleanup temporary files
        try:
            for temp_file in temp_files:
                if temp_file.exists():
                    os.remove(temp_file)
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


@app.route('/OCR/api/validate-only', methods=['POST'])
def validate_only():
    """
    Validation-only endpoint (no OCR)
    Returns validation results without processing cells
    """
    temp_files = []
    
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400
        
        uploaded_file = request.files['file']
        
        if uploaded_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Save file
        temp_path = TEMP_FOLDER / f"{uuid.uuid4().hex}_{uploaded_file.filename}"
        uploaded_file.save(str(temp_path))
        temp_files.append(temp_path)
        
        # Run validation only (debug=False)
        validator = SHGImageValidator(debug=False)
        validation_result = validator.validate_image(str(temp_path))
        
        # Make JSON-safe
        validation_result = make_json_serializable(validation_result)
        
        return jsonify({
            'success': True,
            'validation': validation_result
        })
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        
    finally:
        for temp_file in temp_files:
            if temp_file.exists():
                os.remove(temp_file)


@app.route('/OCR/api/financial/upload', methods=['POST'])
def upload_financial_file():
    """Upload Excel file for SHG financial analytics (shared with frontend dashboards)."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Please upload an Excel file (.xlsx or .xls)."
            }), 400

        file_path = FINANCIAL_FOLDER / "financial_data.xlsx"
        file.save(str(file_path))

        logger.info(f"Financial analytics file uploaded: {file_path}")
        return jsonify({
            "success": True,
            "message": "Financial data uploaded successfully",
            "path": str(file_path)
        })
    except Exception as e:
        logger.error(f"Financial upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/OCR/api/financial/data', methods=['GET'])
def get_financial_data():
    """Serve aggregated SHG financial analytics identical to legacy app.py behavior."""
    try:
        file_path = FINANCIAL_FOLDER / "financial_data.xlsx"
        if not file_path.exists():
            return jsonify({
                "success": False,
                "error": "No financial data found. Please upload Excel file.",
                "data": None
            }), 404

        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()

        district = request.args.get('district')
        mandal = request.args.get('mandal')
        village = request.args.get('village')
        year = request.args.get('year')
        month = request.args.get('month')

        filtered_df = df.copy()

        if district and 'District' in filtered_df:
            filtered_df = filtered_df[filtered_df['District'].astype(str).str.lower() == district.lower()]
        if mandal and 'Mandal' in filtered_df:
            filtered_df = filtered_df[filtered_df['Mandal'].astype(str).str.lower() == mandal.lower()]
        if village and 'Village' in filtered_df:
            filtered_df = filtered_df[filtered_df['Village'].astype(str).str.lower() == village.lower()]
        if year and 'Year' in filtered_df:
            filtered_df = filtered_df[filtered_df['Year'].astype(str) == str(year)]
        if month and 'Month' in filtered_df:
            filtered_df = filtered_df[filtered_df['Month'].astype(str).str.lower() == str(month).lower()]

        if filtered_df.empty:
            return jsonify({
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
            })

        def column_sum(frame, column_name):
            """Strict column sum by exact column name."""
            if column_name not in frame.columns:
                return 0.0
            return float(pd.to_numeric(frame[column_name], errors='coerce').fillna(0).sum())

        def flexible_column_sum(frame, candidates):
            """
            Sum values from the first matching column using a list of possible
            column names. Matching is case-insensitive and ignores extra spaces.
            """
            if frame is None or frame.empty:
                return 0.0

            normalized_cols = {
                str(col).strip().lower(): col for col in frame.columns
            }

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
            # Savings
            "Total Savings Balance": ["Total Savings Balance"],
            "This Month Savings": ["This Month Savings"],

            # Loan balances (totals)
            "Total - SHG Loan Balance": ["Total - SHG Loan Balance"],
            "Total - Bank Loan Balance": ["Total - Bank Loan Balance"],
            "Total - Streenidhi Micro Loan Balance": [
                "Total - Streenidhi Micro Loan Balance",
                "Total Streenidhi Micro Loan Balance",
                "Total - Srinidi Micro Loan Balance",
                "Total Srinidi Micro Loan Balance",
                "Total Srinidhi Micro Loan Balance",
            ],
            "Total - Streenidhi Tenny Loan Balance": [
                "Total - Streenidhi Tenny Loan Balance",
                "Total Streenidhi Tenny Loan Balance",
                "Total - Srinidi Tenny Loan Balance",
                "Total Srinidi Tenny Loan Balance",
                "Total Srinidhi Tenny Loan Balance",
            ],
            "Total - Unnathi SCSP Loan Balance": [
                "Total - Unnathi SCSP Loan Balance",
                "Total Unnathi SCSP Loan Balance",
                "Total - Unnati SCSP Loan Balance",
                "Total Unnati SCSP Loan Balance",
            ],
            "Total - Unnathi TSP Loan Balance": [
                "Total - Unnathi TSP Loan Balance",
                "Total Unnathi TSP Loan Balance",
                "Total - Unnati TSP Loan Balance",
                "Total Unnati TSP Loan Balance",
            ],
            "Total - CIF Loan Balance": [
                "Total - CIF Loan Balance",
                "Total CIF Loan Balance",
            ],
            "Total - VO Loan Balance": ["Total - VO Loan Balance"],

            # This month repayments (for trends + district snapshot)
            "This Month SHG Paid Loan": ["This Month SHG Paid Loan"],
            "This Month Bank Loan Paid": ["This Month Bank Loan Paid"],
            "This Month Streenidhi Micro Loan Paid": [
                "This Month Streenidhi Micro Loan Paid",
                "This Month Srinidi Micro Loan Paid",
                "This Month Srinidhi Micro Loan Paid",
            ],
            "This Month Streenidhi Tenny Loan Paid": [
                "This Month Streenidhi Tenny Loan Paid",
                "This Month Srinidi Tenny Loan Paid",
                "This Month Srinidhi Tenny Loan Paid",
            ],
            "This Month Unnathi SCSP Loan Paid": [
                "This Month Unnathi SCSP Loan Paid",
                "This Month Unnati SCSP Loan Paid",
            ],
            "This Month Unnathi TSP Loan Paid": [
                "This Month Unnathi TSP Loan Paid",
                "This Month Unnati TSP Loan Paid",
            ],
            "This Month CIF Loan Paid": ["This Month CIF Loan Paid"],
            "This Month VO Loan Paid": ["This Month VO Loan Paid"],

            # New loans / other flows
            "New Total": ["New Total"],

            # Current month extras
            "Penalties": [
                "Penalties",
                "Penalty",
                "This Month Penalties",
                "This Month Penalty",
                "Penalty Amount",
            ],
            "Entry Membership Fee": [
                "Entry Membership Fee",
                "Membership Entry Fee",
                "This Month Membership Entry Fee",
                "This Month Entry Membership Fee",
            ],
            "Savings Returned": [
                "Savings Returned",
                "This Month Savings Returned",
                "Returned Savings",
            ],
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
                    logger.info(f"Found loan type column: '{col}' (matched as 'New Loan Type')")
                    break

        if new_loan_type_col:
            logger.info(f"Using column '{new_loan_type_col}' for loan type distribution")
            try:
                type_counts = (
                    filtered_df[new_loan_type_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .replace("", np.nan)
                    .dropna()
                    .value_counts()
                )
                loan_type_distribution = [
                    {"name": str(name), "value": int(count)}
                    for name, count in type_counts.items()
                    if str(name).strip() and str(name).lower() not in ['unknown', 'nan', '']
                ]
                loan_type_distribution.sort(key=lambda x: x['value'], reverse=True)
                logger.info(f"Loan type distribution calculated: {len(loan_type_distribution)} types found")
            except Exception as e:
                logger.error(f"Error calculating loan type distribution: {e}")
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

        repayment_trends = []
        for display_name, canonical_key in repayment_defs:
            value = column_sum_with_synonyms(filtered_df, canonical_key)
            repayment_trends.append({"name": display_name, "paid": value})

        district_shg_loans = []
        district_savings = []
        district_new_loans = []
        district_summaries = {}

        CUSTOM_TOTAL_COLUMNS = [
            # Balances
            "Total Savings Balance",
            "Total - SHG Loan Balance",
            "Total - Bank Loan Balance",
            "Total - Streenidhi Micro Loan Balance",
            "Total - Streenidhi Tenny Loan Balance",
            "Total - Unnathi SCSP Loan Balance",
            "Total - Unnathi TSP Loan Balance",
            "Total - CIF Loan Balance",
            "Total - VO Loan Balance",

            # New loans (totals)
            "New Total",

            # This month flows (for Current Month view)
            "This Month Savings",
            "This Month SHG Paid Loan",
            "This Month Bank Loan Paid",
            "This Month Streenidhi Micro Loan Paid",
            "This Month Streenidhi Tenny Loan Paid",
            "This Month Unnathi SCSP Loan Paid",
            "This Month Unnathi TSP Loan Paid",
            "This Month CIF Loan Paid",
            "This Month VO Loan Paid",

            # Current-month extras
            "Penalties",
            "Entry Membership Fee",
            "Savings Returned",
        ]

        if 'District' in filtered_df.columns:
            all_districts = filtered_df['District'].dropna().astype(str).str.strip().unique()
            all_districts = [d for d in all_districts if d and d.lower() != 'nan']

            for district_name in all_districts:
                group = filtered_df[filtered_df['District'].astype(str).str.strip() == district_name]

                shg_balance = column_sum(group, 'Total - SHG Loan Balance')
                if shg_balance > 0:
                    district_shg_loans.append({
                        "name": district_name,
                        "value": shg_balance
                    })

                savings_balance = column_sum(group, 'Total Savings Balance')
                if savings_balance > 0:
                    district_savings.append({
                        "name": district_name,
                        "value": savings_balance
                    })

                new_loan_count = 0
                if new_loan_type_col and new_loan_type_col in group.columns:
                    try:
                        new_loan_count = (
                            group[new_loan_type_col]
                            .astype(str)
                            .str.strip()
                            .replace('', np.nan)
                            .replace('nan', np.nan)
                            .notna()
                            .sum()
                        )
                    except Exception as e:
                        logger.error(f"Error calculating new loan count for district {district_name}: {e}")
                        new_loan_count = 0
                elif 'New Loan Type' in group.columns:
                    try:
                        new_loan_count = (
                            group['New Loan Type']
                            .astype(str)
                            .str.strip()
                            .replace('', np.nan)
                            .replace('nan', np.nan)
                            .notna()
                            .sum()
                        )
                    except Exception as e:
                        logger.error(f"Error calculating new loan count for district {district_name}: {e}")
                        new_loan_count = 0

                district_new_loans.append({
                    "name": district_name,
                    "value": int(new_loan_count)
                })

                mandal_count = int(
                    group["Mandal"].dropna().astype(str).str.strip().str.lower().nunique()
                ) if "Mandal" in group.columns else 0

                village_count = int(
                    group["Village"].dropna().astype(str).str.strip().str.lower().nunique()
                ) if "Village" in group.columns else 0

                column_totals = {}
                for canonical_key in CUSTOM_TOTAL_COLUMNS:
                    amount = column_sum_with_synonyms(group, canonical_key)
                    column_totals[canonical_key] = amount

                district_summaries[district_name] = {
                    "district": district_name,
                    "forms": int(len(group)),
                    "mandals": mandal_count,
                    "villages": village_count,
                    "savings_total": float(savings_balance),
                    "column_totals": column_totals,
                }

            district_new_loans.sort(key=lambda x: x['name'])
            district_shg_loans.sort(key=lambda x: x['name'])
            district_savings.sort(key=lambda x: x['name'])

            logger.info(f"District aggregations: {len(district_new_loans)} districts with new loan counts")

        logger.info(f"Returning financial data: loan_type_distribution={len(loan_type_distribution)} types, district_new_loans={len(district_new_loans)} districts")

        return jsonify({
            "success": True,
            "data": {
                "savings": savings,
                "loan_portfolio": loan_portfolio,
                "loan_type_distribution": loan_type_distribution,
                "repayment_trends": repayment_trends,
                "district_shg_loans": district_shg_loans,
                "district_savings": district_savings,
                "district_new_loans": district_new_loans,
                "district_summaries": district_summaries
            }
        })
    except Exception as e:
        logger.error(f"Financial data fetch error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/OCR/api/analytics/upload', methods=['POST'])
def upload_analytics_file():
    """Upload Excel file for Data Capture Analytics"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Please upload an Excel file (.xlsx or .xls)."
            }), 400

        file_path = ANALYTICS_FOLDER / "analytics_data.xlsx"
        file.save(str(file_path))

        logger.info(f"Analytics data file uploaded: {file_path}")
        return jsonify({
            "success": True,
            "message": "Analytics data uploaded successfully",
            "path": str(file_path)
        })
    except Exception as e:
        logger.error(f"Analytics upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/OCR/api/analytics/load-sample', methods=['POST'])
def load_sample_analytics():
    """Load sample analytics file if it exists"""
    try:
        sample_file = ANALYTICS_FOLDER / "Sample Data SHG Data Capture.xlsx"
        target_file = ANALYTICS_FOLDER / "analytics_data.xlsx"
        
        if sample_file.exists():
            import shutil
            shutil.copy2(str(sample_file), str(target_file))
            logger.info(f"Sample analytics file loaded: {target_file}")
            return jsonify({
                "success": True,
                "message": "Sample analytics data loaded successfully",
                "path": str(target_file)
            })
        else:
            return jsonify({
                "success": False,
                "error": "Sample file not found"
            }), 404
    except Exception as e:
        logger.error(f"Sample load error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/OCR/api/analytics/data', methods=['GET'])
def get_analytics_data():
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
            return jsonify({
                "success": False,
                "error": "No analytics data found. Please ensure Excel file is in analytics_data folder.",
                "data": None
            }), 404

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "success": False,
                "error": f"Error reading Excel file: {str(e)}",
                "data": None
            }), 500
        
        if df.empty:
            return jsonify({
                "success": False,
                "error": "Excel file is empty",
                "data": None
            }), 400
        
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

        # Get filter parameters
        district = request.args.get('district')
        mandal = request.args.get('mandal')
        village = request.args.get('village')
        year = request.args.get('year')
        month = request.args.get('month')

        filtered_df = df.copy()

        # Apply filters with found column names
        try:
            if district and district_col:
                filtered_df = filtered_df[filtered_df[district_col].astype(str).str.lower() == district.lower()]
            if mandal and mandal_col:
                filtered_df = filtered_df[filtered_df[mandal_col].astype(str).str.lower() == mandal.lower()]
            if village and village_col:
                filtered_df = filtered_df[filtered_df[village_col].astype(str).str.lower() == village.lower()]
            if year and year_col:
                filtered_df = filtered_df[filtered_df[year_col].astype(str) == str(year)]
            if month and month_col:
                filtered_df = filtered_df[filtered_df[month_col].astype(str).str.lower() == str(month).lower()]
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            logger.error(traceback.format_exc())
            # Continue with unfiltered data if filter fails

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
        
        # Get validation failed details - records where Validation Failed > 0
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
                        # Replace NaN values with None (which becomes null in JSON) or empty string
                        details_df = failed_records[available_cols].fillna('')
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
        
        return jsonify({
            "success": True,
            "data": {
                "summary": clean_summary,
                "validation_failed_details": validation_failed_details[:100],  # Limit to 100 records
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
        })
    except Exception as e:
        logger.error(f"Analytics data fetch error: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e)
        # Provide more helpful error message
        if "No such file" in error_message or "does not exist" in error_message:
            error_message = "Analytics Excel file not found. Please ensure the file is in the analytics_data folder."
        return jsonify({
            "success": False, 
            "error": error_message,
            "data": None
        }), 500


@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Get all districts"""
    try:
        districts = [
            {
                "id": index + 1,
                "name": name
            }
            for index, name in enumerate(districts_cache)
        ]
        
        return jsonify({
            "success": True,
            "count": len(districts),
            "districts": districts
        })
    except Exception as e:
        logger.error(f"Error fetching districts: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/mandals', methods=['GET'])
def get_mandals():
    """Get mandals for a specific district"""
    try:
        district_name = request.args.get('district')
        
        if not district_name:
            return jsonify({
                "success": False,
                "error": "District parameter is required"
            }), 400
        
        # Find matching district (case-insensitive)
        matching_district = None
        for dist in districts_cache:
            if dist.lower() == district_name.lower():
                matching_district = dist
                break
        
        if not matching_district:
            return jsonify({
                "success": False,
                "error": f"District '{district_name}' not found"
            }), 404
        
        mandals = mandals_by_district_cache.get(matching_district, [])
        
        district_id = districts_cache.index(matching_district) + 1
        mandals_with_ids = [
            {
                "id": district_id * 1000 + index + 1,
                "name": name,
                "districtId": district_id,
                "districtName": matching_district
            }
            for index, name in enumerate(mandals)
        ]
        
        return jsonify({
            "success": True,
            "district": matching_district,
            "districtId": district_id,
            "count": len(mandals_with_ids),
            "mandals": mandals_with_ids
        })
    except Exception as e:
        logger.error(f"Error fetching mandals: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/villages', methods=['GET'])
def get_villages():
    """Get villages for a specific mandal"""
    try:
        district_name = request.args.get('district')
        mandal_name = request.args.get('mandal')
        
        if not district_name or not mandal_name:
            return jsonify({
                "success": False,
                "error": "Both district and mandal parameters are required"
            }), 400
        
        # Find matching district (case-insensitive)
        matching_district = None
        for dist in districts_cache:
            if dist.lower() == district_name.lower():
                matching_district = dist
                break
        
        if not matching_district:
            return jsonify({
                "success": False,
                "error": f"District '{district_name}' not found"
            }), 404
        
        # Find matching mandal (case-insensitive)
        mandals = mandals_by_district_cache.get(matching_district, [])
        matching_mandal = None
        for mandal in mandals:
            if mandal.lower() == mandal_name.lower():
                matching_mandal = mandal
                break
        
        if not matching_mandal:
            return jsonify({
                "success": False,
                "error": f"Mandal '{mandal_name}' not found in district '{matching_district}'"
            }), 404
        
        key = f"{matching_district}|{matching_mandal}"
        villages = villages_by_mandal_cache.get(key, [])
        
        # If no villages in CSV, return mandal name as default village
        if not villages:
            villages = [matching_mandal]
        
        district_id = districts_cache.index(matching_district) + 1
        mandal_id = mandals.index(matching_mandal) + 1
        
        villages_with_ids = [
            {
                "id": (district_id * 1000 + mandal_id) * 1000 + index + 1,
                "name": name,
                "mandalId": district_id * 1000 + mandal_id,
                "mandalName": matching_mandal,
                "districtId": district_id,
                "districtName": matching_district
            }
            for index, name in enumerate(villages)
        ]
        
        return jsonify({
            "success": True,
            "district": matching_district,
            "mandal": matching_mandal,
            "count": len(villages_with_ids),
            "villages": villages_with_ids
        })
    except Exception as e:
        logger.error(f"Error fetching villages: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({
        "success": False,
        "error": str(e)
    }), 500


if __name__ == '__main__':
    print("=" * 70)
    print("SHG Form OCR Processing System - Google Vision")
    print("=" * 70)
    print("Features:")
    print("  ✓ Image validation pipeline")
    print("  ✓ Table structure detection")
    print("  ✓ Cell-by-cell extraction")
    print("  ✓ Google Vision text recognition")
    print("  ✓ PDF support" if PDF_SUPPORT else "  ⚠ PDF support (install PyMuPDF)")
    print("  ✓ Multi-file processing")
    print("=" * 70)
    print("Endpoints:")
    print("  Health:   http://localhost:5002/OCR/api/health")
    print("  Extract:  http://localhost:5002/OCR/api/extract-tables")
    print("  Validate: http://localhost:5002/OCR/api/validate-only")
    print("  Districts: http://localhost:5002/api/districts")
    print("  Mandals:   http://localhost:5002/api/mandals?district=DistrictName")
    print("  Villages: http://localhost:5002/api/villages?district=DistrictName&mandal=MandalName")
    print("=" * 70)
    print("\n🚀 Starting Flask server with CPU optimizations...")
    print("📊 Watch for throughput metrics (cells/sec) in logs\n")
    app.run(host='0.0.0.0', port=5002, debug=True)