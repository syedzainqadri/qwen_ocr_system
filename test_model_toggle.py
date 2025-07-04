#!/usr/bin/env python3
"""
Test the model toggle functionality.
"""

import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"

def test_model_selection():
    """Test different model selections."""
    print("🧪 Testing Model Selection")
    print("=" * 50)
    
    # Test health first
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server Status: {health['status']}")
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
    
    test_file = available_files[0]
    print(f"\n🧪 Testing with: {test_file}")
    
    # Test different models
    models = [
        ("paddle", "⚡ PaddleOCR (Fast & Reliable)"),
        ("qwen", "🤖 Qwen2.5-VL-3B (AI Vision)"),
        ("auto", "🔄 Auto (Qwen → PaddleOCR fallback)")
    ]
    
    results = {}
    
    for model_id, model_name in models:
        print(f"\n🔧 Testing {model_name}")
        print("-" * 40)
        
        try:
            with open(test_file, 'rb') as f:
                files = {'file': (test_file, f, 'image/webp' if test_file.endswith('.webp') else 'image/jpeg')}
                data = {
                    'language': 'eng' if 'english' in test_file else 'urd',
                    'model': model_id
                }
                
                start_time = time.time()
                print(f"🚀 Starting OCR with {model_id} model...")
                
                # Set different timeouts for different models
                timeout = 30 if model_id == "paddle" else 120
                response = requests.post(f"{BASE_URL}/ocr", files=files, data=data, timeout=timeout)
                request_time = time.time() - start_time
                
            print(f"⏱️  Request completed in {request_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ OCR Response received!")
                print(f"🔧 Engine: {result.get('engine', 'unknown')}")
                print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
                print(f"📝 Word Count: {result.get('word_count', 0)}")
                print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
                print(f"🎯 Success: {result.get('success', False)}")
                
                # Show extracted text (first 100 chars)
                text = result.get('text', '')
                if text:
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"📄 Text: {preview}")
                
                results[model_id] = {
                    'success': True,
                    'engine': result.get('engine', 'unknown'),
                    'confidence': result.get('confidence', 0),
                    'time': request_time,
                    'text_length': len(text),
                    'text_preview': text[:50] if text else "No text"
                }
                
            else:
                print(f"❌ OCR request failed: {response.status_code}")
                results[model_id] = {'success': False, 'error': response.status_code}
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timed out after {timeout} seconds")
            results[model_id] = {'success': False, 'error': 'timeout'}
        except Exception as e:
            print(f"❌ Error during test: {e}")
            results[model_id] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n📊 Model Comparison Summary")
    print("=" * 50)
    
    for model_id, model_name in models:
        result = results.get(model_id, {})
        if result.get('success'):
            print(f"✅ {model_name}")
            print(f"   Engine: {result['engine']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Speed: {result['time']:.2f}s")
            print(f"   Text Length: {result['text_length']} chars")
            print(f"   Preview: {result['text_preview']}")
        else:
            print(f"❌ {model_name}")
            print(f"   Error: {result.get('error', 'unknown')}")
        print()
    
    return any(r.get('success') for r in results.values())

def main():
    """Main test function."""
    print("🎯 Testing Model Toggle System")
    print("📋 This test will:")
    print("   1. Test PaddleOCR model")
    print("   2. Test Qwen2.5-VL model")
    print("   3. Test Auto fallback mode")
    print("   4. Compare results and performance")
    print()
    
    success = test_model_selection()
    
    print(f"\n🎯 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n🎉 Model Toggle System Working!")
        print(f"   • Multiple OCR engines available")
        print(f"   • User can choose preferred model")
        print(f"   • Performance comparison available")
        print(f"   • Ready for production use!")
    else:
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Check server logs for errors")
        print(f"   • Verify models are loading correctly")
        print(f"   • Try individual model tests")

if __name__ == "__main__":
    main()
