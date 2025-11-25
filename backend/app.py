from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google.cloud import vision
from google.oauth2 import service_account
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from PIL import Image, ImageOps
import cv2
import io
import json
import traceback
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('flask_app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

FINANCIAL_FOLDER = (Path(__file__).resolve().parent / 'financial_data')
FINANCIAL_FOLDER.mkdir(parents=True, exist_ok=True)

# Replace with your Vision API credentials
SERVICE_ACCOUNT_JSON = {
  "type": "service_account",
  "project_id": "outstanding-yew-476610-h5",
  "private_key_id": "5c4801e0229e5a9252422d799a8d2b96ac93e0fe",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDpdpqn6jGoaay+\nCTkDsB7Eg7m0lqIOizjh4pS4Rwu7iKDkQH1cLRjHiWJI5ThGFteJ+bc4tApF29qS\n1CqIpRK2mlx5YJ6NYapEEY5G9z16W90ERUeuJ7YdSytg+OkBhb1L68oHWLFGLivy\nSnBURkqqEFfR/u+tq35LfkKmAnR0tEBs0AHW0Tl8hxXvPV5HQxbVHqoKHuJJzIZ6\nQYwqT5NKVz7/SSYTNy0ajPmrc5OIcF5KtaK9e/aJ54tXXhwZgp/0W3kDO4hov1V2\ngerCRGOmX3ktK0ndcgOaHryidLxt0PuZ83TSP5vpF7Qya7NsL8S8tM2zt9WcggFf\nltmBZp0PAgMBAAECggEAVbt+tqvrWWeZDl4vqDmMSuj+kKECWOnqVRUSPQul9NOL\nFdbS0j8jSt8aDx/RxvdLZnkjvfhrj1TZkrLD/dL0qMbNr6r5/nw/fOifgVL4qg7C\n/nb8iClAGMjKYL13P15f8dngIkuBKf75l2ubjW8Uqxf+T/jZBkMkSU/P5Mug/Vui\ngWYp23OW9XVBDzpVl1xhciefkuNR9lmZs1hfmDPNvAraaLxmQNtjS9S6HIyYbvdy\nyYdvA0A8H4L3s+ZaJMt/m9xwLRdiOYHWaoM8VdpUPfOWNNohAjn9veU2UUYQLPuR\nvsnNDyqMvfKVME7xBce0hwgKRGeTbgyml+hgzIL1mQKBgQD0tZIBVH/6Pkumi8Ot\nK9EdKJ+NmLIUbdaTv4ag05Qt7eVk2b3iJ9jZGveP0LyfGr11+l9vGmKG9/EG6RH4\nObpo4MFpZk2pOnsmXCv3h3vT1BFP/bNb+KL6rnQIdFdT0RNwys6ngLxxzPlUgnJf\nb+oxPXDRO7gWs2w8ETpnRv749wKBgQD0PDMyezpuZFZv/ATnFsEnwPY1P7aV1i3C\nzzdzFaOINc88j57AtPfmpUzbmakkgK1HF2jQ/F6OwXvQpHYf6mFrGAwymo5z6/gp\nKeFg9OaRfRJBf1Q2CAy8eOcSrQB8lUaa9/PuWkFF/BC5jTMW/qbvY6qp8iyVUVbd\nDu6NGU9OqQKBgQDxz86Eg/Sm2xI0dF4bbHYKo07vRBmNOHDWtWca25jMvg11V/lc\nVtXgy9Yghjst2eWohI3zoxYDm1TQ6FV6fcknxBk7xv0tIf35jRFhW79QNnoZGnE3\ni/25S9SbWiPFTpAwYChPu1X7+nnTOcg9aMD6gWVPTPz/abOls7yLu8tPRwKBgG4d\npkegX5veCUq8Kcm27KdrzJX9f+jWhBNNMgblPrHu8NyxNDZWYV7QMHLiOOyIR5fB\n6jQvVMKwYY6UV93T4tBSK021eXyya1TD2SXJxRrbdRuquOETiAqByE0XSxzggNDl\n8kkI0F0pZLEEBIDdl45fNVciJQ+9eJh6Xvum6abhAoGBALs9BflGMxxXbwIqCH9+\nnNa7Evpdz6OIWVgdhlZGGsdcGKj0qwupDk8oK5twMg4iXZsmGvFsJ4ZblLGnkv8B\nWrE8H6jdysdb0Muxlv6dsNODEvCCGE3qlGW0Uir3vSjnQQOyGRxaIqU/D72p/w//\nN9qjnf+6O06FgLNSSAz65hEf\n-----END PRIVATE KEY-----\n",
  "client_email": "ocr-758@outstanding-yew-476610-h5.iam.gserviceaccount.com",
  "client_id": "116253747489591616579",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/ocr-758%40outstanding-yew-476610-h5.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}


try:
    credentials = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_JSON)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    logger.info("Vision API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Vision API client: {e}")
    vision_client = None

def fix_image_orientation(image_bytes):
    """Fix image orientation based on EXIF data."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        original_format = image.format
        logger.debug(f"Image format: {original_format}, size: {image.size}")
        
        try:
            image = ImageOps.exif_transpose(image)
            logger.debug("Applied EXIF transpose")
        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"EXIF transpose failed: {e}, trying manual rotation")
            try:
                exif = image.getexif()
                if exif is not None:
                    orientation = exif.get(274)
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                    logger.debug(f"Applied manual rotation for orientation: {orientation}")
            except Exception as inner_e:
                logger.debug(f"Manual rotation failed: {inner_e}")
        
        output = io.BytesIO()
        if original_format and original_format.upper() in ['PNG', 'JPEG', 'JPG']:
            image.save(output, format=original_format)
        else:
            image.save(output, format='JPEG', quality=95)
        
        output.seek(0)
        result = output.getvalue()
        logger.debug(f"Fixed image size: {len(result)} bytes")
        return result
        
    except Exception as e:
        logger.error(f"Image orientation fix failed: {e}")
        logger.error(traceback.format_exc())
        return image_bytes


def enhance_handwritten_image(image_bytes):
    """Improve contrast and readability for handwritten Telugu text."""
    try:
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            return image_bytes

        height, width = image.shape[:2]
        scale_factor = 2.0 if max(height, width) < 2000 else 1.5
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
        inverted = cv2.bitwise_not(binary)
        final_image = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        success, buffer = cv2.imencode('.png', final_image)
        if not success:
            return image_bytes
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Handwriting enhancement failed: {e}")
        return image_bytes

def get_text_from_word(word):
    """Extract text from a word object"""
    try:
        return ''.join([symbol.text for symbol in word.symbols])
    except Exception as e:
        logger.error(f"Error extracting text from word: {e}")
        return ""

def is_handwritten_text(word):
    """Detect if text is handwritten based on characteristics"""
    try:
        if not word.symbols:
            return False
        
        text = get_text_from_word(word)
        if not text or not text.strip():
            return False
        
        text_lower = text.lower().strip()
        
        header_keywords = [
            'క్రమ', 'uid', 'సభ్యురాలు', 'పొదుపు', 'అప్పు', 'కట్టిన', 'మొత్తం', 'నిల్వ',
            'loan', 'bank', 'vo', 'cif', 'type', 'amount', 'serial', 'member', 'details',
            'స్త్రీనిధి', 'ఉన్నతి', 'కొత్త', 'మంజూరు', 'వసూళ్లు', 'చెల్లింపులు',
            'shg', 'బ్యాంక్', 'సమావేశం', 'జరిగిన', 'గత', 'నెల', 'నిల్వలు', 'అంతర్గత',
            'రికవరీ', 'వివరములు', 'క్యాపిటల్', 'ఖాతా', 'తేదీ', 'నాటికి'
        ]
        
        for kw in header_keywords:
            if kw in text_lower and len(text_lower) <= len(kw) + 5:
                return False
        
        has_numbers = bool(re.search(r'\d', text))
        is_mostly_numbers = bool(re.match(r'^[\d\s.,/-]+$', text))
        
        if is_mostly_numbers or (has_numbers and len(text.strip()) > 0):
            return True
        
        has_telugu = bool(re.search(r'[\u0C00-\u0C7F]', text))
        if has_telugu:
            return True
        
        return len(text.strip()) > 0
        
    except Exception as e:
        logger.debug(f"Error in is_handwritten_text: {e}")
        return True

def extract_words_with_positions(page, filter_handwritten=True):
    """Extract words with detailed position information"""
    words_data = []
    try:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    try:
                        vertices = word.bounding_box.vertices
                        x_coords = [v.x for v in vertices]
                        y_coords = [v.y for v in vertices]
                        text = get_text_from_word(word)

                        if not text.strip():
                            continue

                        if filter_handwritten:
                            if not is_handwritten_text(word):
                                continue

                        words_data.append({
                            'text': text.strip(),
                            'x_min': min(x_coords),
                            'x_max': max(x_coords),
                            'y_min': min(y_coords),
                            'y_max': max(y_coords),
                            'x_center': (min(x_coords) + max(x_coords)) / 2,
                            'y_center': (min(y_coords) + max(y_coords)) / 2,
                            'width': max(x_coords) - min(x_coords),
                            'height': max(y_coords) - min(y_coords)
                        })
                    except Exception as e:
                        logger.error(f"Error processing word: {e}")
                        continue

        logger.debug(f"Extracted {len(words_data)} words from page")
        return words_data
    except Exception as e:
        logger.error(f"Error in extract_words_with_positions: {e}")
        return []

def split_table_sections(words_data, all_words_data=None):
    """Split into main data table and reference table at bottom"""
    try:
        if not words_data:
            return [], []

        words_sorted = sorted(words_data, key=lambda w: w['y_center'])
        y_positions = [w['y_center'] for w in words_sorted]
        min_y = min(y_positions)
        max_y = max(y_positions)
        page_height = max_y - min_y

        reference_keywords = ['సమావేశం', 'వసూళ్లు', 'చెల్లింపులు', 'జరిగిన', 'మొత్తం']
        split_y = max_y
        
        for word in words_sorted:
            text_lower = word['text'].lower()
            if any(kw in text_lower for kw in reference_keywords):
                if word['y_center'] > (min_y + page_height * 0.5):
                    split_y = word['y_center'] - 30
                    logger.debug(f"Found reference section at y={split_y}")
                    break
        
        if split_y == max_y and all_words_data:
            all_sorted = sorted(all_words_data, key=lambda w: w['y_center'])
            for word in all_sorted:
                text_lower = word['text'].lower()
                if any(kw in text_lower for kw in reference_keywords):
                    if word['y_center'] > (min_y + page_height * 0.5):
                        split_y = word['y_center'] - 30
                        break
        
        if split_y == max_y:
            split_y = min_y + page_height * 0.7

        main_words = [w for w in words_sorted if w['y_center'] < split_y]
        reference_words = [w for w in words_sorted if w['y_center'] >= split_y]

        logger.info(f"Split: {len(main_words)} main, {len(reference_words)} reference")
        return main_words, reference_words
    except Exception as e:
        logger.error(f"Error in split_table_sections: {e}")
        return words_data, []

def get_main_table_headers():
    """17 column headers for main table"""
    return [
        "క్రమ. సం.", "UID", "సభ్యురాలు పేరు", "ఈ నెల పొదుపు", "ఈ నెల వరకు పొదుపు నిల్వ",
        "SHG అంతర్గత - కట్టిన మొత్తం", "SHG అంతర్గత - అప్పు నిల్వ", "బ్యాంక్ - కట్టిన మొత్తం",
        "బ్యాంక్ - అప్పు నిల్వ", "స్త్రీనిధి/HD - కట్టిన మొత్తం", "స్త్రీనిధి/HD - అప్పు నిల్వ",
        "ఉన్నతి - కట్టిన మొత్తం", "ఉన్నతి - అప్పు నిల్వ", "CIF/VO - కట్టిన మొత్తం",
        "CIF/VO - అప్పు నిల్వ", "కొత్త అప్పు రకం", "కొత్త అప్పు మొత్తం"
    ]

def cluster_rows_by_gap(words_data, min_gap_ratio=0.6):
    """Cluster words into rows by detecting gaps"""
    if not words_data:
        return []
    
    sorted_words = sorted(words_data, key=lambda w: w['y_center'])
    heights = [w['height'] for w in sorted_words]
    median_height = np.median(heights)
    min_gap = median_height * min_gap_ratio
    
    rows = []
    current_row_words = [sorted_words[0]]
    current_row_y = sorted_words[0]['y_center']
    
    for word in sorted_words[1:]:
        gap = word['y_center'] - current_row_y
        if gap < min_gap:
            current_row_words.append(word)
            current_row_y = np.mean([w['y_center'] for w in current_row_words])
        else:
            rows.append(current_row_words)
            current_row_words = [word]
            current_row_y = word['y_center']
    
    if current_row_words:
        rows.append(current_row_words)
    
    return rows

def detect_columns_from_header(header_rows, expected_cols=17):
    """Detect column boundaries from header rows"""
    if not header_rows:
        return None
    
    all_x_positions = []
    for row in header_rows:
        for word in row:
            all_x_positions.append(word['x_min'])
            all_x_positions.append(word['x_max'])
    
    if not all_x_positions:
        return None
    
    min_x = min(all_x_positions)
    max_x = max(all_x_positions)
    x_positions = sorted(set(all_x_positions))
    
    gaps = []
    for i in range(len(x_positions) - 1):
        gap = x_positions[i + 1] - x_positions[i]
        gaps.append((gap, x_positions[i], x_positions[i + 1]))
    
    gaps.sort(reverse=True)
    num_internal_boundaries = min(expected_cols - 1, len(gaps))
    
    boundary_positions = [min_x]
    for gap_size, x_start, x_end in gaps[:num_internal_boundaries]:
        mid = (x_start + x_end) / 2
        boundary_positions.append(mid)
    boundary_positions.append(max_x + 50)
    boundary_positions.sort()
    
    if len(boundary_positions) < 18:
        width = max_x - min_x
        col_width = width / expected_cols
        boundary_positions = [min_x + i * col_width for i in range(expected_cols + 1)]
    elif len(boundary_positions) > 18:
        boundary_positions = boundary_positions[:18]
    
    return boundary_positions

def assign_words_to_grid(rows, col_boundaries):
    """Assign words from rows to grid cells"""
    grid = defaultdict(list)
    
    for row_idx, row_words in enumerate(rows):
        row_words_sorted = sorted(row_words, key=lambda w: w['x_min'])
        
        for word in row_words_sorted:
            col_idx = -1
            word_center = word['x_center']
            
            for i in range(len(col_boundaries) - 1):
                if col_boundaries[i] <= word_center < col_boundaries[i + 1]:
                    col_idx = i
                    break
            
            if col_idx == -1:
                col_idx = 0 if word_center < col_boundaries[0] else len(col_boundaries) - 2
            
            col_idx = max(0, min(col_idx, len(col_boundaries) - 2))
            grid[(row_idx, col_idx)].append(word)
    
    return grid

def build_table_from_grid(grid, num_rows, num_cols):
    """Build table array from grid"""
    table = []
    
    for row_idx in range(num_rows):
        row = []
        for col_idx in range(num_cols):
            cell_words = grid.get((row_idx, col_idx), [])
            cell_words.sort(key=lambda w: w['x_min'])
            cell_text = ' '.join([w['text'] for w in cell_words])
            row.append(cell_text.strip())
        table.append(row)
    
    return table

HEADER_KEYWORDS = [
    'క్రమ', 'uid', 'సభ్యురాలు', 'పొదుపు', 'అప్పు', 'కట్టిన', 'మొత్తం', 'నిల్వ',
    'loan', 'bank', 'vo', 'cif', 'type', 'amount', 'serial', 'member', 'details',
    'స్త్రీనిధి', 'ఉన్నతి', 'కొత్త', 'మంజూరు', 'వసూళ్లు', 'చెల్లింపులు',
    'shg', 'బ్యాంక్', 'సమావేశం', 'జరిగిన', 'గత', 'నెల', 'నిల్వలు', 'అంతర్గత'
]

def is_header_row(row_words):
    """Check if a row is a header row"""
    if not row_words:
        return False
    
    row_text = ' '.join([w['text'].lower() for w in row_words])
    keyword_count = sum(1 for kw in HEADER_KEYWORDS if kw in row_text)
    
    has_many_keywords = keyword_count >= 2
    digit_count = sum(1 for ch in row_text if ch.isdigit())
    alpha_count = sum(1 for ch in row_text if ch.isalpha())
    
    if has_many_keywords and (digit_count == 0 or digit_count / max(alpha_count, 1) < 0.1):
        return True
    
    return False

def identify_header_and_data(rows):
    """Identify which rows are headers vs data"""
    header_end_idx = 2
    
    for idx in range(min(6, len(rows))):
        if is_header_row(rows[idx]):
            header_end_idx = idx
        else:
            row_text = ' '.join([w['text'].lower() for w in rows[idx]])
            has_numbers = bool(re.search(r'\d', row_text))
            keyword_count = sum(1 for kw in HEADER_KEYWORDS if kw in row_text)
            
            if has_numbers and keyword_count < 2:
                if header_end_idx >= 0:
                    break
    
    return header_end_idx

def is_header_like_row(row_words, min_keyword_hits=2):
    """Heuristic check to identify header-like rows"""
    if not row_words:
        return False
    
    row_text = ' '.join([w['text'] for w in row_words]).strip().lower()
    if not row_text:
        return False
    
    keyword_hits = sum(1 for kw in HEADER_KEYWORDS if kw in row_text)
    if keyword_hits >= min_keyword_hits:
        digit_count = sum(1 for ch in row_text if ch.isdigit())
        alpha_count = sum(1 for ch in row_text if ch.isalpha())
        if digit_count == 0 or digit_count / max(alpha_count, 1) < 0.05:
            return True
    
    return False

def filter_header_like_rows(rows):
    """Remove any residual header-like rows"""
    return [row for row in rows if not is_header_like_row(row)]

def clean_cell(text):
    """Clean cell text"""
    if not text:
        return ''
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\|+', '', text)
    return text.strip()

def process_main_table(words_data):
    """Process main data table"""
    if not words_data:
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info("Processing MAIN DATA TABLE")
    logger.info(f"{'='*60}")
    
    rows = cluster_rows_by_gap(words_data, min_gap_ratio=0.5)
    
    if not rows or len(rows) < 4:
        logger.warning("Not enough rows detected")
        return None
    
    logger.info(f"Detected {len(rows)} rows")
    
    header_end_idx = identify_header_and_data(rows)
    header_rows = rows[:header_end_idx + 1]
    data_rows = filter_header_like_rows(rows[header_end_idx + 1:])
    
    if not data_rows:
        logger.warning("No data rows detected")
        return None
    
    logger.info(f"{len(header_rows)} header rows, {len(data_rows)} data rows")
    
    col_boundaries = detect_columns_from_header(header_rows, expected_cols=17)
    
    if not col_boundaries:
        logger.warning("Could not detect columns")
        return None
    
    logger.info(f"Created {len(col_boundaries)-1} column boundaries")
    
    grid = assign_words_to_grid(data_rows, col_boundaries)
    table_data = build_table_from_grid(grid, len(data_rows), 17)
    cleaned_data = [[clean_cell(cell) for cell in row] for row in table_data]
    
    headers = get_main_table_headers()
    df = pd.DataFrame(cleaned_data, columns=headers)
    df = df[df.astype(str).apply(lambda x: x.str.strip().str.len().sum(), axis=1) > 0]
    df = df.fillna('').replace('nan', '').replace('None', '')
    df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Final main table: {len(df)} rows × {len(df.columns)} columns")
    
    return df

def process_reference_table(words_data):
    """Process reference/summary table - 7 columns"""
    if not words_data:
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info("Processing REFERENCE TABLE")
    logger.info(f"{'='*60}")
    
    rows = cluster_rows_by_gap(words_data, min_gap_ratio=0.5)
    
    if not rows:
        logger.warning("No rows detected")
        return None
    
    logger.info(f"Detected {len(rows)} rows")
    
    filtered_rows = [row for row in rows if not is_header_row(row)]
    rows = filtered_rows if filtered_rows else rows
    
    if not rows:
        return None
    
    all_x = []
    for row in rows:
        for word in row:
            all_x.append(word['x_min'])
            all_x.append(word['x_max'])
    
    if not all_x:
        return None
    
    min_x = min(all_x)
    max_x = max(all_x)
    
    # Create 7 columns for reference table
    col_width = (max_x - min_x) / 7
    col_boundaries = [min_x + i * col_width for i in range(8)]
    
    logger.info(f"Created {len(col_boundaries)-1} column boundaries")
    
    grid = assign_words_to_grid(rows, col_boundaries)
    table_data = build_table_from_grid(grid, len(rows), 7)
    cleaned_data = [[clean_cell(cell) for cell in row] for row in table_data]
    
    headers = [f"col_{i+1}" for i in range(7)]
    df = pd.DataFrame(cleaned_data, columns=headers)
    
    df = df.fillna('').replace('nan', '').replace('None', '')
    df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Final reference table: {len(df)} rows × {len(df.columns)} columns")
    
    return df

def create_reference_table_html(ref_df):
    """Create HTML for reference table with 7 columns and complex structure"""
    if ref_df is None or len(ref_df) == 0:
        return ""
    
    table_df = ref_df.copy()
    
    def format_value(value):
        if pd.isna(value):
            return ''
        text = str(value).strip()
        return '' if text in ['', 'nan', 'None'] else text
    
    def get_value(row_idx, col_idx):
        if 0 <= row_idx < len(table_df) and 0 <= col_idx < len(table_df.columns):
            return format_value(table_df.iloc[row_idx, col_idx])
        return ''
    
    total_rows = len(table_df)
    
    # Row labels for columns 2 and 4
    row_labels = [
        "పొదుపు వసూళ్లు మొత్తం",
        "SHG అంతర్గత అప్పు వసూళ్లు మొత్తం",
        "బ్యాంక్ లోన్ వసూళ్లు మొత్తం",
        "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ అప్పు",
        "ఉన్నతి",
        "CIF అప్పు",
        "VO అంతర్గత అప్పు",
        "SHG కు వచ్చిన కొత్త ఋణాలు",
        "Bank వడ్డీ",
        "ఇతర వసూళ్లు",
        "గ్రాంట్స్ (RF/ఇతరములు)",
        "మొత్తం :"
    ]
    
    col_4_labels = [
        "VOకు చెల్లించిన పొదుపులు",
        "VOకు చెల్లించిన SN పొదుపులు",
        "బ్యాంక్ లోన్ ఋణం",
        "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ ఋణం",
        "ఉన్నతి ఋణం కు",
        "CIF ఋణం కు",
        "VO అంతర్గత ఋణం కు",
        "సభ్యులకు ఇచ్చిన కొత్త అప్పులు",
        "Bank సర్వీస్ ఛార్జీలు",
        "VOకు చెల్లించిన సభ్యతవరుసుము",
        "ఈ నెలలో SB A/C నందు జమ చేసిన మొత్తం",
        "మొత్తం :"
    ]
    
    html = []
    html.append('<div class="table-section">')
    html.append('<table class="shg-table reference-table">')
    
    # Header row
    html.append('<thead>')
    html.append('<tr class="header-row-1">')
    html.append('<th colspan="3" class="col-large">ఈ సమావేశం లో జరిగిన వసూళ్లు మొత్తం</th>')
    html.append('<th colspan="2" class="col-large">ఈ సమావేశం లో జరిగిన చెల్లింపులు మొత్తం</th>')
    html.append('<th colspan="2" class="col-large">గత నెల బ్యాంక్ నిల్వలు</th>')
    html.append('</tr>')
    html.append('</thead>')
    
    html.append('<tbody>')
    
    # Row 1: Data with bank details header
    html.append('<tr class="data-row">')
    html.append('<td class="data-cell serial-cell">1</td>')
    html.append(f'<td class="data-cell">{row_labels[0]}</td>')
    html.append(f'<td class="data-cell">{get_value(0, 1)}</td>')
    html.append(f'<td class="data-cell">{col_4_labels[0]}</td>')
    html.append(f'<td class="data-cell">{get_value(0, 3)}</td>')
    html.append('<td class="data-cell">ఖాతా వివరము</td>')
    html.append('<td class="data-cell">తేదీ నాటికి</td>')
    html.append('</tr>')
    
    # Row 2-4: With bank balance details
    bank_details = [
        ("2", row_labels[1], col_4_labels[1], "చేతి నిల్వ"),
        ("3", row_labels[2], col_4_labels[2], "పొదుపు ఖాతా"),
        ("4", row_labels[3], col_4_labels[3], "బ్యాంక్ లోన్ ఖాతా")
    ]
    
    for serial, label2, label4, bank_label in bank_details:
        idx = int(serial) - 1
        html.append('<tr class="data-row">')
        html.append(f'<td class="data-cell serial-cell">{serial}</td>')
        html.append(f'<td class="data-cell">{label2}</td>')
        html.append(f'<td class="data-cell">{get_value(idx, 1)}</td>')
        html.append(f'<td class="data-cell">{label4}</td>')
        html.append(f'<td class="data-cell">{get_value(idx, 3)}</td>')
        html.append(f'<td class="data-cell">{bank_label}</td>')
        html.append(f'<td class="data-cell">{get_value(idx, 5)}</td>')
        html.append('</tr>')
    
    # Row 5: Bank deposit details
    html.append('<tr class="data-row">')
    html.append('<td class="data-cell serial-cell">5</td>')
    html.append(f'<td class="data-cell">{row_labels[4]}</td>')
    html.append(f'<td class="data-cell">{get_value(4, 1)}</td>')
    html.append(f'<td class="data-cell">{col_4_labels[4]}</td>')
    html.append(f'<td class="data-cell">{get_value(4, 3)}</td>')
    html.append('<td colspan="2" class="data-cell">ఈ నెల Bank నగదు జమ వివరాలు</td>')
    html.append('</tr>')
    
    # Row 6: Amount details
    html.append('<tr class="data-row">')
    html.append('<td class="data-cell serial-cell">6</td>')
    html.append(f'<td class="data-cell">{row_labels[5]}</td>')
    html.append(f'<td class="data-cell">{get_value(5, 1)}</td>')
    html.append(f'<td class="data-cell">{col_4_labels[5]}</td>')
    html.append(f'<td class="data-cell">{get_value(5, 3)}</td>')
    html.append('<td class="data-cell">అమౌంట్ రూ.</td>')
    html.append('<td colspan="2" class="data-cell">అక్షరాల</td>')
    html.append('</tr>')
    
    # Row 7: Member name who deposited
    html.append('<tr class="data-row">')
    html.append('<td class="data-cell serial-cell">7</td>')
    html.append(f'<td class="data-cell">{row_labels[6]}</td>')
    html.append(f'<td class="data-cell">{get_value(6, 1)}</td>')
    html.append(f'<td class="data-cell">{col_4_labels[6]}</td>')
    html.append(f'<td class="data-cell">{get_value(6, 3)}</td>')
    html.append('<td colspan="3" class="data-cell">జమ చేసిన సభ్యురాలు పేరు</td>')
    html.append('</tr>')
    
    # Row 8: Member signature who deposited
    html.append('<tr class="data-row">')
    html.append('<td class="data-cell serial-cell">8</td>')
    html.append(f'<td class="data-cell">{row_labels[7]}</td>')
    html.append(f'<td class="data-cell">{get_value(7, 1)}</td>')
    html.append(f'<td class="data-cell">{col_4_labels[7]}</td>')
    html.append(f'<td class="data-cell">{get_value(7, 3)}</td>')
    html.append('<td colspan="3" class="data-cell">జమ చేసిన సభ్యురాలు సంతకం</td>')
    html.append('</tr>')
    
    # Rows 9-11: Merged signature box
    html.append('<tr class="data-row">')
    html.append('<td class="data-cell serial-cell">9</td>')
    html.append(f'<td class="data-cell">{row_labels[8]}</td>')
    html.append(f'<td class="data-cell">{get_value(8, 1)}</td>')
    html.append(f'<td class="data-cell">{col_4_labels[8]}</td>')
    html.append('<td colspan="4" rowspan="3" class="data-cell signature-box">')
    html.append('<div style="text-align: center; padding: 10px;">')
    html.append('<div style="margin-bottom: 30px;">SHG స్టాంప్ & లీడర్స్ సంతకం</div>')
    html.append('<div style="border-top: 1px solid #ccc; padding-top: 10px;">VOA సంతకం</div>')
    html.append('</div>')
    html.append('</td>')
    html.append('</tr>')
    
    html.append('<tr class="data-row">')
    html.append('<td class="data-cell serial-cell">10</td>')
    html.append(f'<td class="data-cell">{row_labels[9]}</td>')
    html.append(f'<td class="data-cell">{get_value(9, 1)}</td>')
    html.append(f'<td class="data-cell">{col_4_labels[9]}</td>')
    html.append('</tr>')
    
    html.append('<tr class="data-row">')
    html.append('<td class="data-cell serial-cell">11</td>')
    html.append(f'<td class="data-cell">{row_labels[10]}</td>')
    html.append(f'<td class="data-cell">{get_value(10, 1)}</td>')
    html.append(f'<td class="data-cell">{col_4_labels[10]}</td>')
    html.append('</tr>')
    
    # Row 12: Total row
    html.append('<tr class="data-row summary-row">')
    html.append('<td class="data-cell serial-cell">12</td>')
    html.append(f'<td class="data-cell summary-cell">{row_labels[11]}</td>')
    html.append(f'<td class="data-cell">{get_value(11, 1)}</td>')
    html.append(f'<td class="data-cell summary-cell">{col_4_labels[11]}</td>')
    html.append(f'<td class="data-cell">{get_value(11, 3)}</td>')
    html.append('<td colspan="3" class="data-cell"></td>')
    html.append('</tr>')
    
    html.append('</tbody>')
    html.append('</table>')
    html.append('</div>')
    
    return ''.join(html)

def create_combined_html(main_df, ref_df):
    """Create HTML with both tables"""
    html = []
    
    html.append('''
    <style>
        .tables-wrapper {
            margin: 30px 0;
            max-height: 75vh;
            overflow-y: auto;
            padding: 10px;
            background-color: #fdfdff;
            border: 1px solid #d4d7e2;
            border-radius: 10px;
        }
        .table-section {
            margin: 0 0 30px;
        }
        .table-section:last-child {
            margin-bottom: 0;
        }
        .shg-table {
            border-collapse: collapse;
            width: 100%;
            font-family: 'Noto Sans Telugu', Arial, sans-serif;
            margin: 0;
            table-layout: auto;
        }
        .shg-table th, .shg-table td {
            border: 1px solid #000;
            padding: 10px 6px;
            text-align: center;
            vertical-align: middle;
            font-size: 13px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
            min-width: 0;
        }
        .header-row-1 {
            background-color: #4472C4;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }
        .header-row-2 {
            background-color: #5B8FDB;
            color: white;
            font-weight: bold;
            font-size: 13px;
        }
        .header-row-3 {
            background-color: #7FADEF;
            color: white;
            font-weight: 600;
            font-size: 12px;
        }
        .data-cell {
            background-color: white;
            font-size: 13px;
        }
        .data-row:nth-child(even) {
            background-color: #f8f9fa;
        }
        .col-small { width: 50px; }
        .col-medium { width: 120px; }
        .col-large { width: 200px; }
        .reference-table th {
            font-size: 13px;
            padding: 10px 8px;
        }
        .reference-table td {
            font-size: 12px;
            padding: 8px 6px;
        }
        .reference-table .data-cell:nth-child(2),
        .reference-table .data-cell:nth-child(4) {
            text-align: left;
            font-weight: 600;
        }
        .serial-cell {
            font-weight: 600;
        }
        .summary-row .summary-cell {
            font-weight: 600;
            text-align: left;
            background-color: #eef2ff;
        }
        .signature-box {
            min-height: 80px;
            vertical-align: middle;
            background-color: #f9f9f9;
        }
    </style>
    ''')
    
    has_main = main_df is not None and len(main_df) > 0
    has_reference = ref_df is not None and len(ref_df) > 0
    
    if not (has_main or has_reference):
        return ''.join(html)
    
    html.append('<div class="tables-wrapper">')
    
    # MAIN TABLE
    if has_main:
        html.append('<div class="table-section">')
        html.append('<table class="shg-table">')
    
        # 3-row header
        html.append('<thead>')
        html.append('<tr>')
        html.append('<th rowspan="3" class="header-row-1 col-small">క్రమ.సం.</th>')
        html.append('<th rowspan="3" class="header-row-1 col-medium">UID</th>')
        html.append('<th rowspan="3" class="header-row-1 col-large">సభ్యురాలు పేరు</th>')
        html.append('<th colspan="2" class="header-row-1">పొదుపు</th>')
        html.append('<th colspan="10" class="header-row-1">అప్పు రికవరీ వివరములు</th>')
        html.append('<th colspan="2" class="header-row-1">కొత్త అప్పు మంజూరు</th>')
        html.append('</tr>')
        
        html.append('<tr>')
        html.append('<th rowspan="2" class="header-row-2">ఈ నెల పొదుపు</th>')
        html.append('<th rowspan="2" class="header-row-2">ఈ నెల వరకు పొదుపు నిల్వ</th>')
        html.append('<th colspan="2" class="header-row-2">SHG అంతర్గత అప్పు</th>')
        html.append('<th colspan="2" class="header-row-2">బ్యాంక్ అప్పు</th>')
        html.append('<th colspan="2" class="header-row-2">స్త్రీనిధి/HD/సీడ్ క్యాపిటల్</th>')
        html.append('<th colspan="2" class="header-row-2">ఉన్నతి</th>')
        html.append('<th colspan="2" class="header-row-2">CIF/VO అంతర్గత</th>')
        html.append('<th rowspan="2" class="header-row-2">అప్పు రకం</th>')
        html.append('<th rowspan="2" class="header-row-2">మొత్తం</th>')
        html.append('</tr>')
        
        html.append('<tr>')
        for _ in range(5):
            html.append('<th class="header-row-3">కట్టిన మొత్తం</th>')
            html.append('<th class="header-row-3">అప్పు నిల్వ</th>')
        html.append('</tr>')
        html.append('</thead>')
    
        # Data rows
        html.append('<tbody>')
        for idx in range(len(main_df)):
            html.append('<tr class="data-row">')
            for col in main_df.columns:
                val = main_df.loc[idx, col]
                display = str(val).strip() if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None'] else ''
                html.append(f'<td class="data-cell">{display}</td>')
            html.append('</tr>')
        html.append('</tbody>')
        html.append('</table>')
        html.append('</div>')
    
    # REFERENCE TABLE
    if has_reference:
        html.append(create_reference_table_html(ref_df))
    
    html.append('</div>')
    
    return ''.join(html)

def process_document(document, filename):
    """Process document - extract both tables"""
    all_tables = []
    
    for page_idx, page in enumerate(document.pages):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Processing page {page_idx + 1} from {filename}")
        logger.info(f"{'#'*80}")
        
        all_words_data = extract_words_with_positions(page, filter_handwritten=False)
        words_data = extract_words_with_positions(page, filter_handwritten=True)
        
        if not words_data:
            logger.warning("No handwritten words detected")
            continue
        
        logger.info(f"Extracted {len(words_data)} handwritten words")
        
        main_words, ref_words = split_table_sections(words_data, all_words_data=all_words_data)
        main_df = process_main_table(main_words) if main_words else None
        ref_df = process_reference_table(ref_words) if ref_words else None
        
        if main_df is None and ref_df is None:
            logger.warning("No tables extracted")
            continue
        
        html_content = create_combined_html(main_df, ref_df)
        
        csv_parts = []
        if main_df is not None:
            csv_parts.append("### MAIN DATA TABLE ###")
            csv_parts.append(main_df.to_csv(index=False, encoding='utf-8-sig'))
        if ref_df is not None:
            csv_parts.append("\n### REFERENCE TABLE ###")
            csv_parts.append(ref_df.to_csv(index=False, encoding='utf-8-sig'))
        
        combined_csv = '\n'.join(csv_parts)
        
        if main_df is not None:
            records = main_df.to_dict(orient='records')
            row_count = len(main_df)
            col_count = len(main_df.columns)
        else:
            records = ref_df.to_dict(orient='records')
            row_count = len(ref_df)
            col_count = len(ref_df.columns)
        
        table_info = {
            "table_id": page_idx,
            "row_count": row_count,
            "col_count": col_count,
            "dataframe": records,
            "csv": combined_csv,
            "json": json.dumps({
                "main": main_df.to_dict(orient='records') if main_df is not None else [], 
                "reference": ref_df.to_dict(orient='records') if ref_df is not None else []
            }, ensure_ascii=False),
            "html": html_content,
            "headers": get_main_table_headers(),
            "has_main_table": main_df is not None,
            "has_reference_table": ref_df is not None
        }
        
        all_tables.append(table_info)
    
    return all_tables

@app.route('/api/health', methods=['GET'])
def health_check():
    status = {
        "status": "healthy",
        "vision_api_initialized": vision_client is not None
    }
    logger.info(f"Health check: {status}")
    return jsonify(status)

@app.route('/api/extract-tables', methods=['POST'])
def extract_tables():
    try:
        logger.info("="*80)
        logger.info("Received extract-tables request")
        
        if vision_client is None:
            logger.error("Vision API client not initialized")
            return jsonify({
                "success": False, 
                "error": "Vision API not configured"
            }), 500
        
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        uploaded_file = request.files['file']
        
        if uploaded_file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        logger.info(f"Processing file: {uploaded_file.filename}")
        
        file_content = uploaded_file.read()
        corrected_content = fix_image_orientation(file_content)
        enhanced_content = enhance_handwritten_image(corrected_content)
        
        image = vision.Image(content=enhanced_content)
        image_context = vision.ImageContext(language_hints=['te', 'en'])
        response = vision_client.document_text_detection(image=image, image_context=image_context)
        
        if response.error.message:
            return jsonify({"success": False, "error": f"Vision API Error: {response.error.message}"}), 500
        
        document = response.full_text_annotation
        
        if not document or not document.pages:
            return jsonify({"success": False, "error": "No text detected"}), 400
        
        tables = process_document(document, uploaded_file.filename)
        
        if not tables:
            return jsonify({"success": False, "error": "No tables extracted"}), 400
        
        logger.info(f"Successfully extracted {len(tables)} table(s)")
        
        return jsonify({
            "success": True,
            "tables": tables,
            "extraction_method": "dual-table-7-column-reference",
            "total_tables": len(tables),
            "total_rows": sum(t.get('row_count', 0) for t in tables)
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/view-table-html', methods=['POST'])
def view_table_html():
    """View tables as formatted HTML"""
    try:
        if 'file' not in request.files:
            return Response("No file uploaded", status=400)
        
        uploaded_file = request.files['file']
        file_content = uploaded_file.read()
        corrected_content = fix_image_orientation(file_content)
        
        image = vision.Image(content=corrected_content)
        response = vision_client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(response.error.message)
        
        document = response.full_text_annotation
        
        if document and document.pages:
            tables = process_document(document, uploaded_file.filename)
            
            if tables and len(tables) > 0:
                html_content = tables[0].get('html', '')
                
                full_html = f'''
                <!DOCTYPE html>
                <html lang="te">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>SHG Tables - {uploaded_file.filename}</title>
                    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;600;700&display=swap" rel="stylesheet">
                    <style>
                        body {{
                            margin: 20px;
                            font-family: 'Noto Sans Telugu', Arial, sans-serif;
                            background-color: #f5f5f5;
                        }}
                        .container {{
                            max-width: 100%;
                            background-color: white;
                            padding: 30px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            border-radius: 8px;
                        }}
                        .page-title {{
                            text-align: center;
                            color: #4472C4;
                            font-size: 24px;
                            font-weight: bold;
                            margin-bottom: 10px;
                            padding-bottom: 10px;
                            border-bottom: 3px solid #4472C4;
                        }}
                        .subtitle {{
                            text-align: center;
                            font-size: 16px;
                            color: #666;
                            margin-bottom: 30px;
                        }}
                        .info-box {{
                            margin-top: 30px;
                            padding: 20px;
                            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                            border-left: 5px solid #4472C4;
                            border-radius: 8px;
                        }}
                        .info-box p {{
                            margin: 8px 0;
                            font-size: 14px;
                            color: #333;
                        }}
                        .info-box strong {{
                            color: #1565c0;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="page-title">
                            స్వయం సహాయక సంఘం - ఆర్థిక లావాదేవీల వివరాలు
                        </div>
                        <div class="subtitle">
                            Self Help Group - Financial Transaction Details
                        </div>
                        
                        {html_content}
                        
                        <div class="info-box">
                            <p><strong>File:</strong> {uploaded_file.filename}</p>
                            <p><strong>Main Table Rows:</strong> {tables[0].get('row_count', 0)}</p>
                            <p><strong>Main Table Columns:</strong> 17 (Fixed)</p>
                            <p><strong>Reference Table Columns:</strong> 7 (Collections, Payments, Bank Balance)</p>
                            <p><strong>Method:</strong> Dual Table System with 7-Column Reference</p>
                        </div>
                    </div>
                </body>
                </html>
                '''
                
                return Response(full_html, mimetype='text/html')
        
        return Response("No tables detected", status=404)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        error_html = f'''
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body style="padding: 20px;">
            <h2 style="color: red;">Error Processing Image</h2>
            <p><strong>Error:</strong> {str(e)}</p>
            <pre style="background: #f5f5f5; padding: 15px;">{traceback.format_exc()}</pre>
        </body>
        </html>
        '''
        return Response(error_html, mimetype='text/html', status=500)


@app.route('/api/financial/upload', methods=['POST'])
def upload_financial_file():
    """Upload Excel file for SHG financial analytics"""
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


@app.route('/api/financial/data', methods=['GET'])
def get_financial_data():
    """Serve aggregated SHG financial analytics"""
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
            if column_name not in frame:
                return 0.0
            return float(pd.to_numeric(frame[column_name], errors='coerce').fillna(0).sum())

        savings = {
            "this_month": column_sum(filtered_df, 'This Month Savings'),
            "total": column_sum(filtered_df, 'Total Savings Balance')
        }

        loan_types = [
            ('SHG Loans', 'Total - SHG Loan Balance'),
            ('Bank Loans', 'Total - Bank Loan Balance'),
            ('Srinidi Micro', 'Total - Srinidi Micro Loan Balance'),
            ('Srinidi Tenny', 'Total - Srinidi Tenny Loan Balance'),
            ('Unnati SCSP', 'Total - Unnati SCSP Loan Balance'),
            ('Unnati TSP', 'Total - Unnati TSP Loan Balance'),
            ('CIF Loans', 'Total - CIF Loan Balance'),
            ('VO Loans', 'Total - VO Loan Balance')
        ]

        loan_portfolio = []
        for name, col in loan_types:
            value = column_sum(filtered_df, col)
            if value > 0:
                loan_portfolio.append({"name": name, "value": value})

        # Loan type distribution (based on New Loan Type column value counts)
        loan_type_distribution = []
        # Try to find the column with flexible matching
        new_loan_type_col = None
        possible_names = ['new loan type', 'newloantype', 'loan type', 'loantype', 'new loan', 'newloan']
        
        # First try exact match
        if 'New Loan Type' in filtered_df.columns:
            new_loan_type_col = 'New Loan Type'
        else:
            # Try case-insensitive and flexible matching
            for col in filtered_df.columns:
                col_lower = col.strip().lower()
                if any(name in col_lower for name in possible_names):
                    new_loan_type_col = col
                    logger.info(f"Found loan type column: '{col}' (matched as 'New Loan Type')")
                    break
        
        if new_loan_type_col:
            logger.info(f"Using column '{new_loan_type_col}' for loan type distribution")
        
        if new_loan_type_col and new_loan_type_col in filtered_df.columns:
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
                # Sort by value descending for better visualization
                loan_type_distribution.sort(key=lambda x: x['value'], reverse=True)
                logger.info(f"Loan type distribution calculated: {len(loan_type_distribution)} types found")
            except Exception as e:
                logger.error(f"Error calculating loan type distribution: {e}")
                loan_type_distribution = []

        repayment_cols = [
            ('SHG', 'This Month SHG Paid Loan'),
            ('Bank', 'This Month Bank Loan Paid'),
            ('Srinidi Micro', 'This Month Srinidi Micro Loan Paid'),
            ('Srinidi Tenny', 'This Month Srinidi Tenny Loan Paid'),
            ('Unnati SCSP', 'This Month Unnati SCSP Loan Paid'),
            ('Unnati TSP', 'This Month Unnati TSP Loan Paid'),
            ('CIF', 'This Month CIF Loan Paid'),
            ('VO', 'This Month VO Loan Paid')
        ]

        repayment_trends = []
        for name, col in repayment_cols:
            value = column_sum(filtered_df, col)
            repayment_trends.append({"name": name, "paid": value})

        # District-wise aggregations for charts and map
        district_shg_loans = []
        district_savings = []
        district_new_loans = []
        district_summaries = {}
        
        CUSTOM_TOTAL_COLUMNS = [
            "Total Savings Balance",
            "Total - SHG Loan Balance",
            "Total - Bank Loan Balance",
            "Total - Srinidi Micro Loan Balance",
            "Total - Srinidi Tenny Loan Balance",
            "Total - Unnati SCSP Loan Balance",
            "Total - Unnati TSP Loan Balance",
            "Total - CIF Loan Balance",
            "Total - VO Loan Balance",
            "New Loan Type",
            "New Total",
        ]
        
        if 'District' in filtered_df.columns:
            # Get all unique districts first to ensure all are included
            all_districts = filtered_df['District'].dropna().astype(str).str.strip().unique()
            all_districts = [d for d in all_districts if d and d.lower() != 'nan']
            
            for district_name in all_districts:
                group = filtered_df[filtered_df['District'].astype(str).str.strip() == district_name]
                
                # SHG Loan Balance by District
                shg_balance = column_sum(group, 'Total - SHG Loan Balance')
                if shg_balance > 0:
                    district_shg_loans.append({
                        "name": district_name,
                        "value": shg_balance
                    })
                
                # Savings Balance by District
                savings_balance = column_sum(group, 'Total Savings Balance')
                if savings_balance > 0:
                    district_savings.append({
                        "name": district_name,
                        "value": savings_balance
                    })
                
                # New Loan Type Count by District (count of entries with New Loan Type)
                # Count non-null and non-empty values - include ALL districts
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
                
                # Always include district, even with 0 count
                district_new_loans.append({
                    "name": district_name,
                    "value": int(new_loan_count)
                })
                
                # Build district summary for map (similar to analytics endpoint)
                mandal_count = int(
                    group["Mandal"].dropna().astype(str).str.strip().str.lower().nunique()
                ) if "Mandal" in group.columns else 0
                
                village_count = int(
                    group["Village"].dropna().astype(str).str.strip().str.lower().nunique()
                ) if "Village" in group.columns else 0
                
                # Calculate column totals for district snapshot
                column_totals = {col: 0 for col in CUSTOM_TOTAL_COLUMNS}
                for col_name in CUSTOM_TOTAL_COLUMNS:
                    if col_name in group.columns:
                        series = group[col_name]
                        numeric_values = pd.to_numeric(series, errors='coerce')
                        if numeric_values.notna().any():
                            amount = float(numeric_values.fillna(0).sum())
                        else:
                            cleaned = series.fillna('').astype(str).str.strip()
                            cleaned = cleaned[cleaned != ""]
                            amount = int(cleaned.count())
                        column_totals[col_name] = amount
                
                district_summaries[district_name] = {
                    "district": district_name,
                    "forms": int(len(group)),
                    "mandals": mandal_count,
                    "villages": village_count,
                    "savings_total": float(savings_balance),
                    "column_totals": column_totals,
                }
            
            # Sort districts alphabetically for consistent display
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
                "loan_type_distribution": loan_type_distribution,  # Based on New Loan Type column
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

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("Enhanced Dual Table System Server Starting...")
    print("=" * 70)
    print("Features:")
    print("  - 17-column main data table")
    print("  - 7-column reference table:")
    print("    * Columns 1-3: Collections section")
    print("    * Columns 4-5: Payments section")
    print("    * Columns 6-7: Bank balance section")
    print("    * Rows 9-11: Merged signature box")
    print("=" * 70)
    print("Endpoints:")
    print("  Health: http://localhost:5000/api/health")
    print("  Extract: http://localhost:5000/api/extract-tables")
    print("  View HTML: http://localhost:5000/api/view-table-html")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=True)