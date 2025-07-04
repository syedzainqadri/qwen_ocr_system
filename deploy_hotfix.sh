#!/bin/bash

# Quick hotfix deployment script

echo "🚨 Deploying Hotfix for Coolify Issues"
echo "====================================="

# Check if we have a remote
if git remote get-url origin >/dev/null 2>&1; then
    echo "📡 Pushing hotfix to GitHub..."
    git push origin main
    
    echo ""
    echo "✅ Hotfix pushed to GitHub!"
    echo ""
    echo "🚀 Next Steps in Coolify:"
    echo "1. Go to your Coolify project dashboard"
    echo "2. Click 'Redeploy' button"
    echo "3. Wait for build to complete (may take 5-10 minutes)"
    echo "4. Check logs for successful startup"
    echo ""
    echo "🔍 Look for these success indicators:"
    echo "   ✅ 'Server starting at http://0.0.0.0:PORT'"
    echo "   ✅ 'Uvicorn running on http://0.0.0.0:PORT'"
    echo "   ✅ 'PaddleOCR Engine initialized'"
    echo "   ✅ No NumPy error messages"
    echo ""
    echo "🧪 Test after deployment:"
    echo "   curl https://your-app.coolify.io/health"
    echo ""
    echo "📚 See HOTFIX_DEPLOYMENT.md for detailed troubleshooting"
    
else
    echo "❌ No GitHub remote found."
    echo ""
    echo "📝 To add GitHub remote:"
    echo "1. Create repository at https://github.com/new"
    echo "2. Run: git remote add origin https://github.com/yourusername/qwen-ocr-system.git"
    echo "3. Run: git push -u origin main"
    echo "4. Then redeploy in Coolify"
fi
