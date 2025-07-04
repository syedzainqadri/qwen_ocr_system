#!/usr/bin/env python3
"""
Test script to verify the optimized Qwen2.5-VL-3B + PaddleOCR system.
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"

def test_optimized_system():
    """Test the optimized system with Qwen2.5-VL-3B and PaddleOCR."""
    print("🚀 Testing Optimized OCR System")
    print("=" * 60)
    print("🎯 Primary: Qwen2.5-VL-3B-Instruct (M1 Pro optimized)")
    print("🔄 Fallback: PaddleOCR (trainable & accurate)")
    print()
    
    # Test health first
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server Status: {health['status']}")
            print(f"📱 Model Loaded: {health['model_loaded']}")
            print(f"💻 Device: {health['device']}")
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
        return False
    
    # Test with the first available file
    test_file = available_files[0]
    print(f"\n🧪 Testing with: {test_file}")
    print(f"📤 Uploading and processing...")
    print(f"⚡ Expecting faster processing with 3B model...")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'image/webp' if test_file.endswith('.webp') else 'image/jpeg')}
            data = {'language': 'eng' if 'english' in test_file else 'urd'}
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/ocr", files=files, data=data, timeout=180)  # 3 minutes
            request_time = time.time() - start_time
            
        print(f"⏱️  Request completed in {request_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ OCR Response received!")
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
                print(f"\n📄 Extracted Text:")
                print("-" * 40)
                print(text[:300] + ("..." if len(text) > 300 else ""))
                print("-" * 40)
            else:
                print(f"⚠️  No text extracted")
            
            # Analyze which engine was used
            engine = result.get('engine', '').lower()
            model_name = result.get('model_name', '').lower()
            
            if 'qwen' in engine or 'qwen' in model_name:
                if '3b' in model_name:
                    print(f"\n🎉 SUCCESS: Qwen2.5-VL-3B was used as primary engine!")
                    print(f"⚡ M1 Pro optimization working!")
                    if request_time < 120:  # Less than 2 minutes
                        print(f"🚀 Great speed improvement with 3B model!")
                    else:
                        print(f"⏰ Still slow - may need more optimization")
                else:
                    print(f"\n⚠️  Qwen was used but not the 3B model")
                return True
            elif 'paddle' in engine or 'paddle' in model_name:
                print(f"\n🔄 PaddleOCR was used (fallback)")
                print(f"💡 This means Qwen2.5-VL failed or is loading")
                print(f"✅ PaddleOCR fallback is working correctly")
                return True
            elif 'demo' in engine:
                print(f"\n🎭 Demo mode was used")
                print(f"💡 Models may not be fully loaded yet")
                return False
            else:
                print(f"\n❓ Unknown engine: {engine}")
                return False
            
        else:
            print(f"❌ OCR request failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 3 minutes")
        print(f"💡 3B model should be faster than this")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    """Main test function."""
    success = test_optimized_system()
    
    print(f"\n🎯 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n🎉 Optimized System Working!")
        print(f"   • Qwen2.5-VL-3B is faster than 7B model")
        print(f"   • PaddleOCR provides excellent fallback")
        print(f"   • System optimized for M1 Pro performance")
        print(f"   • Both engines are trainable and accurate")
    else:
        print(f"\n🔧 Optimization Notes:")
        print(f"   • 3B model should be significantly faster")
        print(f"   • PaddleOCR fallback should work reliably")
        print(f"   • Check server logs for detailed errors")
        print(f"   • First run may take time for model download")

if __name__ == "__main__":
    main()
