#!/bin/bash

# Script to push Qwen OCR System to GitHub

set -e

echo "🚀 Pushing Qwen OCR System to GitHub"
echo "===================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run 'git init' first."
    exit 1
fi

# Check if user provided repository URL
if [ $# -eq 0 ]; then
    echo "📝 Usage: ./push_to_github.sh <github-repo-url>"
    echo ""
    echo "📋 Steps to create GitHub repository:"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: qwen-ocr-system"
    echo "3. Description: Powerful OCR system with Qwen2.5-VL-3B and PaddleOCR"
    echo "4. Make it Public (for Coolify deployment)"
    echo "5. Don't initialize with README (we already have one)"
    echo "6. Copy the repository URL and run:"
    echo "   ./push_to_github.sh https://github.com/yourusername/qwen-ocr-system.git"
    exit 1
fi

REPO_URL=$1

echo "📡 Adding remote origin: $REPO_URL"
git remote add origin $REPO_URL

echo "🔄 Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "🎉 Successfully pushed to GitHub!"
echo "================================="
echo ""
echo "🌐 Your repository is now available at:"
echo "   $REPO_URL"
echo ""
echo "🚀 Next Steps for Coolify Deployment:"
echo "1. Open your Coolify dashboard"
echo "2. Create new project"
echo "3. Connect GitHub repository: $REPO_URL"
echo "4. Set environment variables:"
echo "   - PORT=8001"
echo "   - ENVIRONMENT=production"
echo "   - PYTHONUNBUFFERED=1"
echo "5. Deploy!"
echo ""
echo "🧪 After deployment, test with:"
echo "   python test_cloud_deployment.py https://your-coolify-url.com"
echo ""
echo "📚 Documentation:"
echo "   - README.md: Project overview and usage"
echo "   - DEPLOYMENT.md: Detailed deployment guide"
echo "   - Docker support: Ready for containerized deployment"
echo ""
echo "✨ Features ready for production:"
echo "   🤖 Qwen2.5-VL-3B + PaddleOCR dual engines"
echo "   🔄 Model toggle system"
echo "   📊 Real-time progress tracking"
echo "   🎯 Multi-language support"
echo "   🚀 Training infrastructure"
echo "   🐳 Docker containerization"
echo ""
echo "🎯 Your OCR system is ready for the cloud! 🚀"
