#!/usr/bin/env python3
"""Simple server runner for Qwen2.5-VL OCR system."""

import uvicorn
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        # Get port from environment (for cloud deployment)
        # Coolify and other platforms may set PORT to different values
        # Use 3030 as default since 8001 may not be available on some platforms
        port = int(os.getenv("PORT", 3030))

        # Detect environment more reliably
        environment = os.getenv("ENVIRONMENT", "production")  # Default to production
        # Also check for common cloud environment indicators
        if any(os.getenv(var) for var in ["RAILWAY_ENVIRONMENT", "RENDER", "HEROKU_APP_NAME", "COOLIFY_APP_ID"]):
            environment = "production"

        reload = environment == "development"

        print("🚀 Starting Qwen2.5-VL OCR Server")
        print("=" * 40)
        print("🎯 Primary Engine: PaddleOCR (reliable & fast)")
        print("🔄 Fallback Engine: Qwen2.5-VL-3B-Instruct (memory issues on cloud)")
        print(f"🌐 Environment: {environment}")
        print(f"🔧 Port: {port}")
        print(f"🔄 Reload: {reload}")
        print(f"✅ Server starting at http://0.0.0.0:{port}")
        print(f"📚 API docs available at http://0.0.0.0:{port}/docs")
        print(f"🌐 Web interface at http://0.0.0.0:{port}")
        print()
        print("⏳ Note: First startup may take 5-10 minutes for model download")
        print("🛑 Press Ctrl+C to stop the server")

        # Run the server with production-optimized settings
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=port,
            reload=False,  # Force disable reload in production
            log_level="info",
            workers=1,  # Single worker for model consistency
            timeout_keep_alive=60,  # Longer keep alive
            access_log=False,  # Disable access logs to reduce noise
            server_header=False,  # Disable server header
            date_header=False  # Disable date header
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
