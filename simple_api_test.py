#!/usr/bin/env python3
"""
Simple API test for Qwen OCR system - quick test without large model download.
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"

def test_health():
    """Test server health."""
    print("🔍 Testing server health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"📱 Model Loaded: {data['model_loaded']}")
            print(f"💻 Device: {data['device']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_with_small_image():
    """Test with a small image file."""
    print(f"\n🧪 Testing OCR with small image...")
    
    # Check if we have test images
    test_files = ["english.webp", "urdu.jpg"]
    available_files = [f for f in test_files if Path(f).exists()]
    
    if not available_files:
        print("❌ No test images found")
        return False
    
    # Test with the first available file
    test_file = available_files[0]
    print(f"📁 Using test file: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'image/webp' if test_file.endswith('.webp') else 'image/jpeg')}
            data = {'language': 'eng' if 'english' in test_file else 'urd'}
            
            print(f"📤 Sending OCR request...")
            print(f"⏱️  Timeout set to 60 seconds...")
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/ocr", files=files, data=data, timeout=60)
            request_time = time.time() - start_time
            
        print(f"⏱️  Request completed in {request_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ OCR Response received!")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"📝 Word Count: {result.get('word_count', 0)}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"🖥️  Model: {result.get('model_name', 'unknown')}")
            print(f"💾 Device: {result.get('device', 'unknown')}")
            print(f"✅ Success: {result.get('success', False)}")
            
            # Show extracted text
            text = result.get('text', '')
            if text:
                print(f"\n📄 Extracted Text (first 200 chars):")
                print("-" * 40)
                print(text[:200] + ("..." if len(text) > 200 else ""))
                print("-" * 40)
            else:
                print(f"⚠️  No text extracted")
            
            # Check for demo mode
            if 'demo' in result.get('engine', '').lower():
                print(f"\n🎭 DEMO MODE DETECTED")
                print(f"💡 This means the full model isn't loaded yet")
                print(f"🔄 The system is using fallback/demo responses")
            
            # Check for errors
            if result.get('error'):
                print(f"⚠️  Error reported: {result['error']}")
            
            return True
            
        else:
            print(f"❌ OCR request failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 60 seconds")
        print(f"💡 This might mean the model is downloading in the background")
        return False
    except Exception as e:
        print(f"❌ Error during OCR test: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Simple Qwen OCR API Test")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("❌ Server not healthy, stopping tests")
        return
    
    # Test OCR
    success = test_with_small_image()
    
    print(f"\n🎯 Test Summary:")
    print(f"{'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n💡 Next steps:")
        print(f"   • Try the web interface at http://localhost:8001")
        print(f"   • Upload your test images via the browser")
        print(f"   • Run full tests with: python test_ocr_results.py")
    else:
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Check if model is downloading in server logs")
        print(f"   • Try again in a few minutes")
        print(f"   • Use the web interface instead")

if __name__ == "__main__":
    main()
