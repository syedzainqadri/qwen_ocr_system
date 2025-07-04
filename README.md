# 🤖 Qwen OCR System

A powerful, production-ready OCR (Optical Character Recognition) system combining **Qwen2.5-VL-3B** (AI Vision) and **PaddleOCR** with real-time progress tracking and model selection.

![OCR Demo](https://img.shields.io/badge/OCR-Qwen2.5--VL--3B-blue) ![PaddleOCR](https://img.shields.io/badge/Fallback-PaddleOCR-green) ![Docker](https://img.shields.io/badge/Docker-Ready-blue) ![FastAPI](https://img.shields.io/badge/API-FastAPI-green)

## ✨ Features

- 🤖 **Dual OCR Engines**: Qwen2.5-VL-3B (AI Vision) + PaddleOCR (Fast & Reliable)
- 🔄 **Model Toggle**: Users can choose their preferred OCR engine
- 📊 **Real-time Progress**: Live progress tracking during OCR processing
- 🎯 **Multi-language Support**: English, Urdu, Arabic, French, German, Spanish
- 🚀 **Training Ready**: Train PaddleOCR with custom data for better accuracy
- 🐳 **Docker Ready**: Containerized for easy cloud deployment
- 🌐 **Web Interface**: Clean, user-friendly web UI
- 📚 **REST API**: Complete API with documentation
- ⚡ **Production Ready**: Optimized for cloud deployment

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/qwen-ocr-system.git
cd qwen-ocr-system

# Build and run with Docker
docker build -t qwen-ocr-system .
docker run -p 8001:8001 qwen-ocr-system
```

### Option 2: Local Development

```bash
# Clone and setup
git clone https://github.com/yourusername/qwen-ocr-system.git
cd qwen-ocr-system

# Install dependencies
pip install -r requirements.txt

# Run the server
python run_server.py
```

Visit: http://localhost:8001

## 🌐 Cloud Deployment

### Deploy to Coolify

1. **Connect Repository**: Add this GitHub repo to Coolify
2. **Set Environment Variables**:
   ```
   PORT=8001
   ENVIRONMENT=production
   PYTHONUNBUFFERED=1
   ```
3. **Deploy**: Coolify will automatically build and deploy using Docker

### Other Platforms

- **Railway**: `./deploy-railway.sh`
- **Render**: Use `render.yaml` configuration
- **Google Cloud Run**: See `DEPLOYMENT.md`
- **Azure**: See `DEPLOYMENT.md`

## 🎯 Usage

### Web Interface

1. Open http://localhost:8001
2. Choose your OCR engine:
   - 🤖 **Qwen2.5-VL-3B**: AI-powered vision model
   - ⚡ **PaddleOCR**: Fast and reliable
   - 🔄 **Auto**: Qwen → PaddleOCR fallback
3. Select language
4. Upload image
5. Get results with real-time progress

### API Usage

```bash
# Health check
curl http://localhost:8001/health

# OCR with PaddleOCR
curl -X POST http://localhost:8001/ocr \
  -F "file=@image.jpg" \
  -F "language=eng" \
  -F "model=paddle"

# OCR with Qwen2.5-VL
curl -X POST http://localhost:8001/ocr \
  -F "file=@image.jpg" \
  -F "language=eng" \
  -F "model=qwen"
```

## 🔧 Configuration

### Environment Variables

```bash
PORT=8001                    # Server port
ENVIRONMENT=production       # Environment (development/production)
PYTHONUNBUFFERED=1          # Better logging
TRANSFORMERS_CACHE=/app/.cache  # Model cache location
```

### Model Selection

- **qwen**: Use Qwen2.5-VL-3B only
- **paddle**: Use PaddleOCR only
- **auto**: Try Qwen first, fallback to PaddleOCR

## 🎓 Training PaddleOCR

Improve accuracy with custom training data:

```python
from train_paddleocr import PaddleOCRTrainer

# Initialize trainer
trainer = PaddleOCRTrainer()

# Add training samples
trainer.add_training_sample("image1.jpg", "Ground truth text 1")
trainer.add_training_sample("image2.jpg", "Ground truth text 2")

# Generate training configuration
trainer.create_data_lists()
trainer.generate_training_config()
```

## 📊 Performance

### Qwen2.5-VL-3B
- **Accuracy**: Excellent for complex layouts
- **Speed**: Slower (30-120s depending on hardware)
- **Best for**: Documents, handwriting, complex text

### PaddleOCR
- **Accuracy**: Good, trainable for specific use cases
- **Speed**: Fast (1-10s)
- **Best for**: Clean text, production environments

## 🧪 Testing

```bash
# Test locally
python test_simple.py

# Test model toggle
python test_model_toggle.py

# Test cloud deployment
python test_cloud_deployment.py https://your-app-url.com
```

## 📁 Project Structure

```
qwen_ocr_system/
├── app/
│   ├── main.py              # FastAPI application
│   ├── qwen_ocr.py          # Qwen2.5-VL engine
│   └── paddle_ocr.py        # PaddleOCR engine
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose
├── requirements.txt         # Python dependencies
├── train_paddleocr.py       # Training system
├── DEPLOYMENT.md           # Deployment guide
└── tests/                  # Test scripts
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Qwen Team** for the amazing Qwen2.5-VL model
- **PaddlePaddle** for PaddleOCR
- **FastAPI** for the excellent web framework
- **Hugging Face** for model hosting and transformers

---

**Made with ❤️ for the OCR community**