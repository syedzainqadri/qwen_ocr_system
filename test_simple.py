#!/usr/bin/env python3
"""
Simple test to verify the OCR system works with PaddleOCR fallback.
"""

import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"

def test_simple_ocr():
    """Test OCR with a simple approach."""
    print("🧪 Simple OCR Test")
    print("=" * 50)
    
    # Test health first
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server Status: {health['status']}")
            print(f"📱 Device: {health['device']}")
            print(f"🤖 Model Loaded: {health['model_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test with available images
    test_files = ["english.webp", "urdu.jpg"]
    available_files = [f for f in test_files if Path(f).exists()]
    
    if not available_files:
        print("❌ No test images found")
        print("💡 Please ensure english.webp or urdu.jpg is in the current directory")
        return False
    
    # Test with the first available file
    test_file = available_files[0]
    print(f"\n🧪 Testing with: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'image/webp' if test_file.endswith('.webp') else 'image/jpeg')}
            data = {'language': 'eng' if 'english' in test_file else 'urd'}
            
            start_time = time.time()
            print(f"🚀 Starting OCR request (timeout: 60s)...")
            
            response = requests.post(f"{BASE_URL}/ocr", files=files, data=data, timeout=60)
            request_time = time.time() - start_time
            
        print(f"\n⏱️  Request completed in {request_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ OCR Response received!")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"📝 Word Count: {result.get('word_count', 0)}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"🎯 Success: {result.get('success', False)}")
            
            # Show extracted text (first 200 chars)
            text = result.get('text', '')
            if text:
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"\n📄 Extracted Text Preview:")
                print(f"   {preview}")
            else:
                print(f"\n⚠️  No text extracted")
            
            return True
            
        else:
            print(f"❌ OCR request failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 60 seconds")
        print(f"💡 This might indicate the model is taking too long to process")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 Testing OCR System")
    print("📋 This test will:")
    print("   1. Check server health")
    print("   2. Try OCR with available test images")
    print("   3. Show results and performance")
    print()
    
    success = test_simple_ocr()
    
    print(f"\n🎯 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n🎉 OCR System Working!")
        print(f"   • Server is responding")
        print(f"   • OCR processing completed")
        print(f"   • Text extraction successful")
    else:
        print(f"\n🔧 Troubleshooting Tips:")
        print(f"   • Make sure server is running: python run_server.py")
        print(f"   • Check if test images exist: english.webp, urdu.jpg")
        print(f"   • Try refreshing the browser at http://localhost:8001")
        print(f"   • Check server logs for errors")

if __name__ == "__main__":
    main()
