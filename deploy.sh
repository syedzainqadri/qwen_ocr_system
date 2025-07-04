#!/bin/bash

# Qwen OCR System - Cloud Deployment Script
# Supports Railway, Render, Heroku, and other platforms

set -e

echo "🚀 Qwen OCR System - Cloud Deployment"
echo "====================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t qwen-ocr-system .

# Test the image locally first
echo "🧪 Testing Docker image locally..."
docker run --rm -d -p 8001:8001 --name qwen-ocr-test qwen-ocr-system

# Wait for the service to start
echo "⏳ Waiting for service to start..."
sleep 30

# Test health endpoint
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Local test successful!"
else
    echo "❌ Local test failed!"
    docker logs qwen-ocr-test
    docker stop qwen-ocr-test
    exit 1
fi

# Stop test container
docker stop qwen-ocr-test

echo "🎉 Docker image built and tested successfully!"
echo ""
echo "📋 Next Steps for Cloud Deployment:"
echo ""
echo "🚂 Railway:"
echo "   1. Install Railway CLI: npm install -g @railway/cli"
echo "   2. Login: railway login"
echo "   3. Deploy: railway up"
echo ""
echo "🎨 Render:"
echo "   1. Connect your GitHub repository to Render"
echo "   2. Create a new Web Service"
echo "   3. Use Docker runtime"
echo "   4. Set port to 8001"
echo ""
echo "🟣 Heroku:"
echo "   1. Install Heroku CLI"
echo "   2. heroku create your-app-name"
echo "   3. heroku container:push web"
echo "   4. heroku container:release web"
echo ""
echo "☁️ Google Cloud Run:"
echo "   1. gcloud builds submit --tag gcr.io/PROJECT-ID/qwen-ocr"
echo "   2. gcloud run deploy --image gcr.io/PROJECT-ID/qwen-ocr --platform managed"
echo ""
echo "🔧 Environment Variables to Set:"
echo "   - PORT=8001"
echo "   - PYTHONUNBUFFERED=1"
echo "   - TRANSFORMERS_CACHE=/app/.cache"
echo ""
echo "💾 Recommended Resources:"
echo "   - Memory: 4-8GB"
echo "   - CPU: 2-4 cores"
echo "   - Storage: 10GB+ (for models)"
echo ""
echo "🌐 After deployment, your OCR system will be available at:"
echo "   https://your-app-url.com"
