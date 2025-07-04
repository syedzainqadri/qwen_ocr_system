#!/usr/bin/env python3
"""Simple server runner for Qwen2.5-VL OCR system."""

import uvicorn
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Get port from environment (for cloud deployment)
    port = int(os.getenv("PORT", 8001))

    # Disable reload in production
    environment = os.getenv("ENVIRONMENT", "development")
    reload = environment == "development"

    print("🚀 Starting Qwen2.5-VL OCR Server")
    print("=" * 40)
    print("🎯 Primary Engine: Qwen2.5-VL-3B-Instruct")
    print("🔄 Fallback Engine: PaddleOCR (trainable & accurate)")
    print(f"🌐 Environment: {environment}")
    print(f"🔧 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"✅ Server starting at http://0.0.0.0:{port}")
    print(f"📚 API docs available at http://0.0.0.0:{port}/docs")
    print(f"🌐 Web interface at http://0.0.0.0:{port}")
    print()
    print("🛑 Press Ctrl+C to stop the server")

    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=port,
            reload=reload,
            log_level="info",
            workers=1  # Single worker for model consistency
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)
