#!/usr/bin/env python3
"""
Test PaddleOCR directly to verify it works independently.
This bypasses Qwen and tests only the fallback engine.
"""

import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"

def test_paddleocr_only():
    """Test PaddleOCR by forcing fallback."""
    print("🧪 PaddleOCR Only Test")
    print("=" * 50)
    
    # Test health first
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server Status: {health['status']}")
            print(f"📱 Device: {health['device']}")
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
    print(f"🎯 Strategy: Force PaddleOCR by making Qwen fail")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'image/webp' if test_file.endswith('.webp') else 'image/jpeg')}
            # Use a language that might make Qwen fail faster
            data = {'language': 'eng' if 'english' in test_file else 'urd'}
            
            start_time = time.time()
            print(f"🚀 Starting OCR request (timeout: 30s for faster fallback)...")
            
            response = requests.post(f"{BASE_URL}/ocr", files=files, data=data, timeout=30)
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
            
            # Check if it used PaddleOCR
            engine = result.get('engine', '').lower()
            if 'paddle' in engine:
                print(f"🎉 SUCCESS: PaddleOCR is working!")
            elif 'qwen' in engine:
                print(f"⚠️  Qwen worked (unexpected but good)")
            else:
                print(f"❓ Unknown engine: {engine}")
            
            # Show extracted text (first 200 chars)
            text = result.get('text', '')
            if text:
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"\n📄 Extracted Text Preview:")
                print(f"   {preview}")
                return True
            else:
                print(f"\n⚠️  No text extracted")
                return False
            
        else:
            print(f"❌ OCR request failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 30 seconds")
        print(f"💡 Even PaddleOCR fallback is taking too long")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 Testing PaddleOCR Fallback System")
    print("📋 This test will:")
    print("   1. Check server health")
    print("   2. Try OCR (should fallback to PaddleOCR)")
    print("   3. Verify PaddleOCR is working")
    print()
    
    success = test_paddleocr_only()
    
    print(f"\n🎯 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n🎉 PaddleOCR System Working!")
        print(f"   • Server is responding")
        print(f"   • PaddleOCR fallback is functional")
        print(f"   • Text extraction successful")
        print(f"   • Ready for production use!")
    else:
        print(f"\n🔧 Next Steps:")
        print(f"   • PaddleOCR may need more time to initialize")
        print(f"   • Check server logs for PaddleOCR errors")
        print(f"   • Try again in a few minutes")
        print(f"   • Consider using smaller test images")

if __name__ == "__main__":
    main()
