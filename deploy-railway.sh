#!/bin/bash

# Quick Railway Deployment Script for Qwen OCR System

set -e

echo "🚂 Railway Deployment for Qwen OCR System"
echo "========================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway:"
    railway login
fi

# Create new Railway project
echo "🆕 Creating new Railway project..."
railway new

# Set environment variables
echo "🔧 Setting environment variables..."
railway variables set PORT=8001
railway variables set ENVIRONMENT=production
railway variables set PYTHONUNBUFFERED=1

# Deploy the application
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "🎉 Deployment Complete!"
echo "========================================"
echo "Your OCR system is now deployed to Railway!"
echo ""
echo "📋 Next Steps:"
echo "1. Check deployment status: railway status"
echo "2. View logs: railway logs"
echo "3. Open your app: railway open"
echo ""
echo "🌐 Your app will be available at:"
echo "   https://your-app-name.railway.app"
echo ""
echo "🧪 Test your deployment:"
echo "   curl https://your-app-name.railway.app/health"
