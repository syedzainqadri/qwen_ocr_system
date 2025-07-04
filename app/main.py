"""FastAPI server for Qwen2.5-VL OCR system."""

import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json as json_lib

try:
    from .qwen_ocr_robust import robust_qwen_ocr
except Exception as e:
    print(f"Failed to import robust_qwen_ocr: {e}")
    robust_qwen_ocr = None

from .paddle_ocr import PaddleOCREngine

# Initialize OCR engines
paddle_ocr = PaddleOCREngine(use_angle_cls=True, lang='en')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket connection manager for progress tracking
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_progress(self, message: str, progress: int):
        """Send progress update to all connected clients."""
        data = {"message": message, "progress": progress}
        for connection in self.active_connections:
            try:
                await connection.send_text(json_lib.dumps(data))
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Create FastAPI app
app = FastAPI(
    title="Qwen2.5-VL OCR System",
    description="Production OCR system powered by Qwen2.5-VL-3B-Instruct with PaddleOCR fallback",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Response models
class OCRResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    language: str
    engine: str
    word_count: int
    processing_time: float
    model_name: str
    device: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Qwen2.5-VL OCR System</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .content {
                padding: 40px;
            }
            
            .upload-section {
                background: #f8f9fa;
                border: 3px dashed #dee2e6;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 30px;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .upload-section:hover {
                border-color: #667eea;
                background: #f0f2ff;
            }
            
            .upload-section.dragover {
                border-color: #667eea;
                background: #e3f2fd;
                transform: scale(1.02);
            }
            
            .upload-icon {
                font-size: 4em;
                color: #667eea;
                margin-bottom: 20px;
            }
            
            .controls {
                display: grid;
                grid-template-columns: 1fr 1fr auto;
                gap: 20px;
                margin-bottom: 30px;
                align-items: end;
            }
            
            .form-group {
                display: flex;
                flex-direction: column;
            }
            
            .form-group label {
                font-weight: 600;
                margin-bottom: 8px;
                color: #333;
            }
            
            .form-group select,
            .form-group input {
                padding: 12px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            
            .form-group select:focus,
            .form-group input:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                height: fit-content;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            
            .results {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                margin-top: 30px;
                display: none;
            }
            
            .results.show {
                display: block;
                animation: slideIn 0.5s ease;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .result-text {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                max-height: 300px;
                overflow-y: auto;
            }
            
            .metadata {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            
            .metadata-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            
            .metadata-item .label {
                font-weight: 600;
                color: #666;
                font-size: 0.9em;
            }
            
            .metadata-item .value {
                font-size: 1.1em;
                color: #333;
                margin-top: 5px;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 40px;
            }
            
            .loading.show {
                display: block;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }
            
            #fileInput {
                display: none;
            }
            
            .file-info {
                margin-top: 15px;
                padding: 10px;
                background: #e3f2fd;
                border-radius: 8px;
                display: none;
            }
            
            .file-info.show {
                display: block;
            }

            .progress-container {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                margin-top: 30px;
                display: none;
            }

            .progress-container.show {
                display: block;
                animation: slideIn 0.5s ease;
            }

            .progress-bar {
                width: 100%;
                height: 20px;
                background-color: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 15px;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                width: 0%;
                transition: width 0.3s ease;
                border-radius: 10px;
            }

            .progress-text {
                text-align: center;
                font-weight: 600;
                color: #495057;
                margin-bottom: 10px;
            }

            .progress-message {
                text-align: center;
                color: #6c757d;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Qwen2.5-VL OCR System</h1>
                <p>Advanced text extraction using Qwen2.5-VL-7B-Instruct model</p>
            </div>
            
            <div class="content">
                <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <h3>Drop your image here or click to browse</h3>
                    <p>Supports: JPG, PNG, WEBP, TIFF, PDF</p>
                    <input type="file" id="fileInput" accept="image/*,.pdf" onchange="handleFileSelect(event)">
                    <div class="file-info" id="fileInfo"></div>
                </div>
                
                <div class="controls">
                    <div class="form-group">
                        <label for="language">Language</label>
                        <select id="language">
                            <option value="eng">English</option>
                            <option value="urd">Urdu</option>
                            <option value="ara">Arabic</option>
                            <option value="fra">French</option>
                            <option value="deu">German</option>
                            <option value="spa">Spanish</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="model">OCR Engine</label>
                        <select id="model">
                            <option value="qwen">🤖 Qwen2.5-VL-3B (AI Vision)</option>
                            <option value="paddle">⚡ PaddleOCR (Fast & Reliable)</option>
                            <option value="auto">🔄 Auto (Qwen → PaddleOCR fallback)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="confidence">Min Confidence</label>
                        <input type="number" id="confidence" value="50" min="0" max="100">
                    </div>
                    
                    <button class="btn" onclick="processImage()" id="processBtn" disabled>
                        🚀 Extract Text
                    </button>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <h3>Processing image with Qwen2.5-VL...</h3>
                    <p>This may take a few moments</p>
                </div>

                <div class="progress-container" id="progressContainer">
                    <h3>📊 Processing Progress</h3>
                    <div class="progress-text" id="progressText">0%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <div class="progress-message" id="progressMessage">Initializing...</div>
                </div>

                <div class="results" id="results">
                    <h3>📄 Extracted Text</h3>
                    <div class="result-text" id="resultText"></div>
                    
                    <div class="metadata" id="metadata"></div>
                </div>
            </div>
        </div>
        
        <script>
            let selectedFile = null;
            let ws = null;

            // Initialize WebSocket connection for progress updates
            function initWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;

                ws = new WebSocket(wsUrl);

                ws.onopen = function() {
                    console.log('WebSocket connected for progress updates');
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateProgress(data.message, data.progress);
                };

                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    // Reconnect after 3 seconds
                    setTimeout(initWebSocket, 3000);
                };

                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }

            // Update progress bar and message
            function updateProgress(message, progress) {
                const progressContainer = document.getElementById('progressContainer');
                const progressFill = document.getElementById('progressFill');
                const progressText = document.getElementById('progressText');
                const progressMessage = document.getElementById('progressMessage');

                progressFill.style.width = progress + '%';
                progressText.textContent = progress + '%';
                progressMessage.textContent = message;

                // Show progress container when processing starts
                if (progress > 0 && !progressContainer.classList.contains('show')) {
                    progressContainer.classList.add('show');
                }

                // Hide progress container when complete
                if (progress >= 100) {
                    setTimeout(() => {
                        progressContainer.classList.remove('show');
                    }, 2000);
                }
            }

            // Initialize WebSocket on page load
            initWebSocket();
            
            // Drag and drop functionality
            const uploadSection = document.querySelector('.upload-section');
            
            uploadSection.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadSection.classList.add('dragover');
            });
            
            uploadSection.addEventListener('dragleave', () => {
                uploadSection.classList.remove('dragover');
            });
            
            uploadSection.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });
            
            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    handleFile(file);
                }
            }
            
            function handleFile(file) {
                selectedFile = file;
                const fileInfo = document.getElementById('fileInfo');
                fileInfo.innerHTML = `
                    <strong>Selected:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
                    <strong>Type:</strong> ${file.type}
                `;
                fileInfo.classList.add('show');
                document.getElementById('processBtn').disabled = false;
            }
            
            async function processImage() {
                if (!selectedFile) {
                    alert('Please select an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('language', document.getElementById('language').value);
                formData.append('model', document.getElementById('model').value);
                
                // Show loading and reset progress
                document.getElementById('loading').classList.add('show');
                document.getElementById('results').classList.remove('show');
                document.getElementById('progressContainer').classList.remove('show');
                document.getElementById('processBtn').disabled = true;

                // Reset progress
                updateProgress('Starting OCR process...', 0);
                
                try {
                    const response = await fetch('/ocr', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        displayResults(result);
                    } else {
                        displayError(result.error || 'OCR processing failed');
                    }
                } catch (error) {
                    displayError('Network error: ' + error.message);
                } finally {
                    document.getElementById('loading').classList.remove('show');
                    document.getElementById('processBtn').disabled = false;
                    // Progress container will auto-hide after completion
                }
            }
            
            function displayResults(result) {
                document.getElementById('resultText').textContent = result.text || 'No text detected';
                
                const metadata = document.getElementById('metadata');
                metadata.innerHTML = `
                    <div class="metadata-item">
                        <div class="label">Confidence</div>
                        <div class="value">${result.confidence.toFixed(1)}%</div>
                    </div>
                    <div class="metadata-item">
                        <div class="label">Language</div>
                        <div class="value">${result.language}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="label">Engine</div>
                        <div class="value">${result.engine}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="label">Word Count</div>
                        <div class="value">${result.word_count}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="label">Processing Time</div>
                        <div class="value">${result.processing_time.toFixed(2)}s</div>
                    </div>
                    <div class="metadata-item">
                        <div class="label">Device</div>
                        <div class="value">${result.device}</div>
                    </div>
                `;
                
                document.getElementById('results').classList.add('show');
            }
            
            function displayError(error) {
                const results = document.getElementById('results');
                results.innerHTML = `
                    <div class="error">
                        <h3>❌ Error</h3>
                        <p>${error}</p>
                    </div>
                `;
                results.classList.add('show');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=robust_qwen_ocr is not None and robust_qwen_ocr.model_loaded,
        device=robust_qwen_ocr.device if robust_qwen_ocr else "unavailable"
    )

@app.post("/ocr", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(...),
    language: str = Form("eng"),
    model: str = Form("auto")
):
    """Extract text from uploaded image using Qwen2.5-VL."""
    
    # Validate file type
    allowed_types = {
        "image/jpeg", "image/jpg", "image/png",
        "image/tiff", "image/webp", "application/pdf",
        "application/octet-stream"  # Allow generic binary files
    }

    # Also check file extension if content type is generic
    if file.content_type == "application/octet-stream":
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".pdf"}
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension: {file_extension}"
            )
    elif file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"Processing file: {file.filename} ({file.content_type})")
        
        # Create progress callback for real-time updates
        def progress_callback(message: str, progress: int):
            # Simple synchronous callback that just prints progress
            # The WebSocket updates will be handled separately
            print(f"📊 Progress: {progress}% - {message}")

            # Store progress in a simple way that can be accessed by WebSocket
            if hasattr(progress_callback, 'last_progress'):
                progress_callback.last_progress = (message, progress)
            else:
                progress_callback.last_progress = (message, progress)

        # Model selection based on user choice
        logger.info(f"User selected model: {model}")

        if model == "paddle":
            logger.info("User selected: PaddleOCR only")
            try:
                result = paddle_ocr.extract_text(str(file_path), language, progress_callback)
            except Exception as e:
                logger.error(f"PaddleOCR failed: {e}")
                result = {"success": False, "error": str(e), "text": "", "confidence": 0.0}
        elif model == "qwen":
            logger.info("User selected: Qwen2.5-VL only (with timeout protection)")
            if robust_qwen_ocr is None:
                logger.error("Qwen2.5-VL not available")
                result = {"success": False, "error": "Qwen2.5-VL not available", "text": "", "confidence": 0.0}
            else:
                try:
                    result = robust_qwen_ocr.extract_text(str(file_path), language, progress_callback)
                except Exception as e:
                    logger.error(f"Qwen2.5-VL failed: {e}")
                    result = {"success": False, "error": str(e), "text": "", "confidence": 0.0}
        else:  # model == "auto"
            logger.info("Auto mode: Qwen2.5-VL → PaddleOCR fallback")
            # Use Qwen2.5-VL-3B as primary OCR engine, PaddleOCR as fallback
            if robust_qwen_ocr is None:
                logger.info("Qwen2.5-VL not available, using PaddleOCR only...")
                result = paddle_ocr.extract_text(str(file_path), language, progress_callback)
            else:
                try:
                    logger.info("Trying Qwen2.5-VL-3B first (with timeout protection)...")
                    result = robust_qwen_ocr.extract_text(str(file_path), language, progress_callback)

                    # If Qwen times out, has errors, or fails, fallback to PaddleOCR
                    if (result.get("timeout_occurred") or result.get("error") or
                        not result.get("success", True)):
                        if result.get("timeout_occurred"):
                            logger.info("Qwen2.5-VL timed out, falling back to PaddleOCR...")
                        else:
                            logger.info("Qwen2.5-VL failed, falling back to PaddleOCR...")

                        try:
                            paddle_result = paddle_ocr.extract_text(str(file_path), language, progress_callback)
                            if paddle_result.get("success", True):
                                result = paddle_result
                                logger.info("PaddleOCR fallback successful")
                        except Exception as paddle_error:
                            logger.warning(f"PaddleOCR fallback also failed: {paddle_error}")
                            # Continue with Qwen result (which includes timeout info)

                except Exception as e:
                    logger.error(f"Qwen2.5-VL failed: {e}")
                    # Fallback to PaddleOCR
                    try:
                        logger.info("Falling back to PaddleOCR due to Qwen error...")
                        result = paddle_ocr.extract_text(str(file_path), language, progress_callback)
                    except Exception as paddle_error:
                        logger.error(f"Both engines failed. Qwen: {e}, PaddleOCR: {paddle_error}")
                        result = {
                            "text": "OCR processing failed",
                            "confidence": 0.0,
                        "language": language,
                        "engine": "error",
                        "word_count": 0,
                        "processing_time": 0.0,
                        "model_name": "none",
                        "device": "none",
                    "error": f"Both engines failed: Qwen: {str(e)}, PaddleOCR: {str(paddle_error)}"
                }
        
        # Clean up uploaded file
        try:
            os.unlink(file_path)
        except:
            pass
        
        # Return structured response
        return OCRResponse(
            success=True,
            text=result.get("text", ""),
            confidence=result.get("confidence", 0.0),
            language=result.get("language", language),
            engine=result.get("engine", "qwen2.5-vl"),
            word_count=result.get("word_count", 0),
            processing_time=result.get("processing_time", 0.0),
            model_name=result.get("model_name", ""),
            device=result.get("device", ""),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return OCRResponse(
            success=False,
            text="",
            confidence=0.0,
            language=language,
            engine="qwen2.5-vl",
            word_count=0,
            processing_time=0.0,
            model_name="",
            device="",
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
