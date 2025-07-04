#!/usr/bin/env python3
"""
Quick demo test for Qwen OCR system - tests demo mode functionality.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8001"

def test_demo_mode():
    """Test the system in demo mode (when models aren't loaded)."""
    print("🚀 Quick Demo Test - Qwen OCR System")
    print("=" * 50)
    
    # Test health first
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server Status: {health['status']}")
            print(f"📱 Model Loaded: {health['model_loaded']}")
            print(f"💻 Device: {health['device']}")
        else:
            print(f"❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test with a small dummy image (create a minimal test file)
    print(f"\n🧪 Testing demo mode functionality...")
    
    # Create a minimal test image file
    test_file = Path("test_demo.txt")
    test_file.write_text("This is a test file for demo mode")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {'language': 'eng'}
            
            print(f"📤 Sending test request...")
            response = requests.post(f"{BASE_URL}/ocr", files=files, data=data, timeout=30)
            
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Response received!")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"📝 Word Count: {result.get('word_count', 0)}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"🖥️  Model: {result.get('model_name', 'unknown')}")
            print(f"💾 Device: {result.get('device', 'unknown')}")
            
            # Check if it's demo mode
            if 'demo' in result.get('engine', '').lower():
                print(f"🎭 Demo Mode Active - Model not fully loaded yet")
                print(f"💡 This is expected during first-time model download")
            
            # Display text
            text = result.get('text', '')
            if text:
                print(f"\n📄 Response Text:")
                print("-" * 30)
                print(text[:200] + ("..." if len(text) > 200 else ""))
                print("-" * 30)
            
            if result.get('error'):
                print(f"⚠️  Error: {result['error']}")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
    
    print(f"\n🎯 Demo test completed!")
    print(f"💡 Once model download completes, you can run full tests with:")
    print(f"   python test_ocr_results.py")

if __name__ == "__main__":
    test_demo_mode()
