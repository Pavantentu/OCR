# OCR Backend

This is the Flask backend service for the OCR application that processes images, PDFs, and DOCX files.

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python app.py
```

The server will start on http://localhost:5000

## API Endpoints

### POST /api/upload
Uploads and processes a file for OCR.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - file: The file to process (image/pdf/docx)
  - language: The language for OCR (default: "en")

**Response:**
```json
{
    "status": "success",
    "text": "Extracted text from the document",
    "error": null
}
```

### GET /api/languages
Returns list of supported languages.

**Response:**
```json
{
    "languages": [
        {"code": "en", "name": "English"},
        {"code": "te", "name": "Telugu"}
    ]
}
```

## Error Handling

The API returns appropriate error messages with HTTP status codes:
- 400: Bad Request (invalid file type, missing file)
- 500: Internal Server Error (processing failed)

Example error response:
```json
{
    "status": "error",
    "text": null,
    "error": "Invalid file type. Please upload an image, PDF, or DOCX file."
}
```