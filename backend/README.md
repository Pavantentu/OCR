# 🔍 SHG OCR Processing System

A production-grade Flask backend service for processing SHG (Self Help Group) handwritten/typed forms with advanced OCR capabilities, comprehensive validation, and financial analytics.

Built with **Flask**, **OpenCV**, **PyMuPDF**, **Google Vision API**, and custom SHG form detection & validation modules.

---

## ✨ Key Features

### 📄 **SHG Form OCR Extraction**
- Supports **image** and **PDF** uploads (multi-page PDF support)
- Automatic PDF-to-image conversion at 300 DPI resolution
- Image validation and format standardization
- Advanced grid/table cell extraction using computer vision
- Dynamic SHG header reconstruction with multi-level headers
- Structured table output with row/column metadata
- **Google Vision OCR** for high-accuracy text recognition
- Automatic SHG MBK ID detection and extraction

### ✅ **Validation Module**
Comprehensive form structure validation including:
- Form orientation detection
- Table grid detection and analysis
- Required header existence verification
- Cell boundary detection
- Detailed validation metadata with confidence scores

### 💰 **Financial Analytics Engine**
- Excel file upload and processing
- **Savings Summary**: This month and total savings tracking
- **Loan Portfolio Analysis**: Multi-type loan balance tracking
- **Loan Type Distribution**: Categorized loan statistics
- **Repayment Trends**: Month-over-month payment tracking
- **District-wise Aggregations**: Geographic breakdowns
- **Synonym-based Column Normalization**: Handles inconsistent Excel column names automatically
- Advanced filtering by district, mandal, village, year, and month

### 📊 **Data Capture Analytics**
Excel-based monitoring dashboard for OCR imports:
- **Total Imports Tracking**: Count of all processed forms
- **Validation Success/Failure Metrics**: Detailed validation statistics
- **MBK Sync Status**: Track synchronization to MBK system
- **District-level Summaries**: Geographic performance metrics
- **Failure Analysis**: Breakdown by error type (incorrect form, missing fields, image quality, etc.)
- **Automatic Column Detection**: Works with varying Excel header formats
- Chart-ready data for visualizations

### 🛠 **Production-Ready Utilities**
- Temporary directory cleanup and management
- Base64 encoding for image data
- JSON-safe serialization for complex data types
- Comprehensive logging (console + UTF-8 file logs)
- CORS configuration for frontend integration
- Error handling and graceful degradation

---

## 📁 Directory Structure

```
backend/
├── app.py                      # Main Flask application
├── validate.py                 # SHG image validation module
├── test.py                     # SHG table extraction & detection module
├── shg_detector/              # SHG detection utilities
│   ├── cell_processing.py
│   ├── cell_tracing.py
│   ├── config.py
│   ├── core.py
│   ├── image_enhancement.py
│   ├── line_detection.py
│   ├── preprocessing.py
│   ├── processor.py
│   ├── table_detection.py
│   ├── training.py
│   └── utils.py
├── uploads/                    # Temporary upload storage
├── temp_processing/           # Runtime temporary files
├── result/                     # Processing results
├── financial_data/            # Financial analytics Excel files
│   └── financial_data.xlsx
├── analytics_data/            # Data capture analytics Excel files
│   ├── analytics_data.xlsx
│   └── Sample Data SHG Data Capture.xlsx
├── flask_app.log              # Application logs (UTF-8 encoded)
├── venv/                      # Python virtual environment
└── README.md                  # This file
```

---

## ⚙️ Installation

### 1. **Create Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `flask` - Web framework
- `flask-cors` - Cross-origin resource sharing
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `PyMuPDF` (fitz) - PDF processing
- `pandas` - Data analysis and Excel handling
- `google-auth` - Google authentication
- `google-auth-oauthlib` - OAuth support
- `pillow` - Image manipulation
- `requests` - HTTP requests for Google Vision API

---

## 🚀 Running the Server

```bash
python app.py
```

The server will start on:
- **URL**: `http://localhost:5002`
- **Health Check**: `http://localhost:5002/OCR/api/health`

---

## 🔐 Configuration

### Google Vision API Setup

The application requires Google Vision API credentials for OCR functionality. Configure in `app.py`:

**Option 1: API Key Mode**
```python
GOOGLE_VISION_API_KEY = "YOUR_API_KEY_HERE"
```

**Option 2: Service Account (Recommended for Production)**
```python
GOOGLE_VISION_API_KEY = {
    "type": "service_account",
    "project_id": "your-project-id",
    "private_key_id": "...",
    "private_key": "...",
    "client_email": "...",
    "client_id": "...",
    "auth_uri": "...",
    "token_uri": "...",
    "auth_provider_x509_cert_url": "...",
    "client_x509_cert_url": "..."
}
```

### CORS Configuration

The app is configured to accept requests from:
- `http://localhost:5173` (local development)
- `https://pavantentu.github.io` (production frontend)

Modify CORS settings in `app.py` if needed.

---

## 🛠 API Endpoints

### 1. **Health Check**

**GET** `/OCR/api/health`

Returns system status and feature availability.

**Response:**
```json
{
  "status": "healthy",
  "pdf_support": true,
  "validation_enabled": true,
  "google_vision_configured": true
}
```

---

### 2. **Extract Tables** (Main Endpoint)

**POST** `/OCR/api/extract-tables`

Processes uploaded images or PDFs, extracts table data, and performs OCR.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` or `files[]`: Image file(s) or PDF file(s)
  - Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.pdf`

**Response Structure:**
```json
{
  "success": true,
  "total_files": 1,
  "total_pages_processed": 1,
  "successful_pages": 1,
  "files": [
    {
      "filename": "form.jpg",
      "file_type": "image",
      "total_pages": 1,
      "success": true,
      "pages": [
        {
          "success": true,
          "page": 1,
          "total_pages": 1,
          "validation": { ... },
          "total_cells": 85,
          "cells": [ ... ],
          "table_data": {
            "shg_mbk_id": "MS-12345",
            "total_rows": 5,
            "total_columns": 17,
            "column_headers": [ ... ],
            "header_rows": [ ... ],
            "data_rows": [
              {
                "row_number": 1,
                "row_index": 0,
                "cells": [
                  {
                    "col_index": 0,
                    "key": "member_mbk_id",
                    "label": "సభ్యురాలి MBK ID",
                    "text": "MBK-001",
                    "confidence": 0.95
                  },
                  ...
                ]
              }
            ]
          }
        }
      ]
    }
  ]
}
```

**Features:**
- Multi-file upload support
- PDF multi-page processing
- Automatic cell extraction and OCR
- Structured table data with row/column mapping
- Validation metadata included

---

### 3. **Validate Only**

**POST** `/OCR/api/validate-only`

Performs validation without OCR processing (faster for validation checks).

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Single image file

**Response:**
```json
{
  "success": true,
  "validation": {
    "is_valid": true,
    "orientation_correct": true,
    "has_grid": true,
    "cell_count": 85,
    ...
  }
}
```

---

### 4. **Financial Analytics - Upload**

**POST** `/OCR/api/financial/upload`

Uploads an Excel file for financial analytics processing.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Excel file (`.xlsx` or `.xls`)

**Response:**
```json
{
  "success": true,
  "message": "Financial data uploaded successfully",
  "path": "financial_data/financial_data.xlsx"
}
```

---

### 5. **Financial Analytics - Get Data**

**GET** `/OCR/api/financial/data?district=...&mandal=...&village=...&year=...&month=...`

Retrieves aggregated financial analytics data with optional filters.

**Query Parameters:**
- `district` (optional): Filter by district name
- `mandal` (optional): Filter by mandal name
- `village` (optional): Filter by village name
- `year` (optional): Filter by year
- `month` (optional): Filter by month

**Response:**
```json
{
  "success": true,
  "data": {
    "savings": {
      "this_month": 125000.00,
      "total": 2450000.00
    },
    "loan_portfolio": [
      {"name": "SHG Loans", "value": 500000.00},
      {"name": "Bank Loans", "value": 750000.00},
      ...
    ],
    "loan_type_distribution": [
      {"name": "Agriculture", "value": 45},
      {"name": "Livestock", "value": 30},
      ...
    ],
    "repayment_trends": [
      {"name": "This Month Savings", "paid": 25000.00},
      {"name": "This Month SHG Paid Loan", "paid": 15000.00},
      ...
    ],
    "district_shg_loans": [ ... ],
    "district_savings": [ ... ],
    "district_new_loans": [ ... ],
    "district_summaries": {
      "District Name": {
        "district": "District Name",
        "forms": 150,
        "mandals": 5,
        "villages": 25,
        "savings_total": 500000.00,
        "column_totals": { ... }
      }
    }
  }
}
```

**Note:** The system automatically handles column name variations using synonym matching for robust Excel processing.

---

### 6. **Analytics - Upload**

**POST** `/OCR/api/analytics/upload`

Uploads an Excel file for data capture analytics.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Excel file (`.xlsx` or `.xls`)

**Response:**
```json
{
  "success": true,
  "message": "Analytics data uploaded successfully",
  "path": "analytics_data/analytics_data.xlsx"
}
```

---

### 7. **Analytics - Load Sample**

**POST** `/OCR/api/analytics/load-sample`

Loads the sample analytics file if it exists in the `analytics_data` folder.

**Response:**
```json
{
  "success": true,
  "message": "Sample analytics data loaded successfully",
  "path": "analytics_data/analytics_data.xlsx"
}
```

---

### 8. **Analytics - Get Data**

**GET** `/OCR/api/analytics/data?district=...&mandal=...&village=...&year=...&month=...`

Retrieves data capture analytics with comprehensive summary information.

**Query Parameters:**
- `district` (optional): Filter by district
- `mandal` (optional): Filter by mandal
- `village` (optional): Filter by village
- `year` (optional): Filter by year
- `month` (optional): Filter by month

**Response:**
```json
{
  "success": true,
  "data": {
    "summary": {
      "total_imports": 1000,
      "validation_successful": 850,
      "validation_failed": 150,
      "synced_to_mkb": 830,
      "success_rate": 83.0
    },
    "validation_failed_details": [
      {
        "District": "District Name",
        "Validation Failed": 10,
        "Failed Incorrect Form": 5,
        "Failed Missing Fields": 3,
        "Failed Image Quality": 2,
        ...
      }
    ],
    "district_summaries": {
      "District Name": {
        "district": "District Name",
        "imports": 500,
        "validation_failed": 50,
        "synced_to_mkb": 450,
        "success_rate": 90.0
      }
    },
    "mandal_summaries": { ... },
    "chart_data": {
      "by_district": { ... },
      "by_year": { ... }
    },
    "filters_applied": {
      "district": null,
      "mandal": null,
      "village": null,
      "year": null,
      "month": null
    }
  }
}
```

---

## 🔄 OCR Pipeline Flow

```
┌─────────────┐
│ Upload File │
└──────┬──────┘
       │
       ├─ PDF ──────────┐
       │                 │
       └─ Image ─────────┤
                         │
              ┌──────────▼──────────┐
              │  Convert to Images  │
              │   (300 DPI if PDF)  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Image Validation   │
              │  (Format & Quality) │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  SHG Form Validation│
              │  (Structure Check)  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Cell Extraction    │
              │  (Grid Detection)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Google Vision OCR  │
              │  (Text Recognition) │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Table Structuring  │
              │  (Row/Column Map)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │    JSON Response    │
              └─────────────────────┘
```

---

## 📋 SHG Table Structure

The system extracts structured data from SHG forms with **17 columns**:

1. సభ్యురాలి MBK ID (Member MBK ID)
2. సభ్యురాలు పేరు (Member Name)
3. ఈ నెల పొదుపు (This Month Savings)
4. ఈ నెల వరకు పొదుపు (Total Savings Till Now)
5. SHG అంతర్గత అప్పు కట్టిన మొత్తం (SHG Internal Loan Total)
6. బ్యాంక్ అప్పు కట్టిన మొత్తం (Bank Loan Total)
7. స్త్రీనిధి మైక్రో అప్పు కట్టిన మొత్తం (Streenidhi Micro Loan Total)
8. స్త్రీనిధి టెన్నీ అప్పు కట్టిన మొత్తం (Streenidhi Tenny Loan Total)
9. ఉన్నతి (SCSP) అప్పు కట్టిన మొత్తం (Unnathi SCSP Loan Total)
10. ఉన్నతి (TSP) అప్పు కట్టిన మొత్తం (Unnathi TSP Loan Total)
11. CIF అప్పు కట్టిన మొత్తం (CIF Loan Total)
12. VO అంతర్గత అప్పు కట్టిన మొత్తం (VO Internal Loan Total)
13. అప్పు రకం (Loan Type)
14. మొత్తం (Amount)
15. జరిమానా (Penalty Amount)
16. సభ్యులకు తిరిగి ఇచ్చిన మొత్తం (Returned to Members)
17. సభ్యుల ఇతర పొదుపులు (Other Savings)

The system automatically reconstructs multi-level headers to match the physical form layout.

---

## 📘 Logging

All application logs are written to:
- **Console**: Real-time output with formatted messages
- **File**: `flask_app.log` (UTF-8 encoded for Telugu character support)

**Log Levels:**
- `DEBUG`: Detailed processing information
- `INFO`: General flow and status updates
- `WARNING`: Non-critical issues
- `ERROR`: Error conditions and exceptions

**Logged Information:**
- File upload and processing status
- Validation flow steps and results
- OCR processing progress
- Cell extraction details
- Table reconstruction logs
- API request/response summaries
- Error traces and stack traces

---

## 🛡 Error Handling

The system gracefully handles:

- ❌ **Missing OCR Credentials**: Returns empty text with warnings
- ❌ **Invalid Image Formats**: Returns validation errors
- ❌ **Failed PDF Parsing**: Returns error with details
- ❌ **Cell Extraction Errors**: Continues with partial results
- ❌ **Excel Column Name Variations**: Uses synonym matching
- ❌ **Missing Excel Files**: Returns 404 with helpful messages
- ❌ **Network Errors**: Handles Google Vision API timeouts
- ❌ **Empty Results**: Returns structured empty responses

All errors are logged with full stack traces for debugging.

---

## 🔧 Technical Details

### Image Processing
- **OpenCV** for image manipulation and validation
- **PIL/Pillow** for format conversion
- High-resolution PDF rendering (300 DPI)
- Automatic image standardization to JPEG format

### OCR Technology
- **Google Vision API** for text recognition
- Support for both API key and service account authentication
- Batch processing for multiple cells
- Confidence scores for each recognized text
- Automatic token refresh for service accounts

### Data Processing
- **Pandas** for Excel file handling
- Automatic column detection and normalization
- Flexible filtering and aggregation
- JSON-safe serialization for complex data types

### File Management
- UUID-based temporary file organization
- Automatic cleanup of temporary files and directories
- Organized folder structure for different data types
- Path validation and error handling

---

## 🚨 Important Notes

1. **Google Vision API**: Ensure credentials are properly configured for OCR functionality
2. **PDF Support**: Requires PyMuPDF (`pip install PyMuPDF`)
3. **Excel Files**: Must be in `.xlsx` or `.xls` format
4. **Port Configuration**: Default port is `5002` (changeable in `app.py`)
5. **File Size Limits**: Consider Flask's default file upload limits for large files
6. **Telugu Support**: System handles Telugu characters in forms and logs (UTF-8 encoding)

---

## 📄 License

Internal project — distribution restricted.

---

## 🔗 Related Modules

- `validate.py`: SHG image validation and structure checking
- `test.py`: SHG form detection and table extraction
- `shg_detector/`: Custom detection and processing utilities

---

## 📞 Support

For issues or questions, refer to:
- Application logs: `flask_app.log`
- Health check endpoint: `GET /OCR/api/health`
- Error responses include detailed error messages

---

**Built with ❤️ for SHG data processing and analytics**