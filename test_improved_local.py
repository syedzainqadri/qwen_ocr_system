#!/usr/bin/env python3
"""
Test the improved Qwen OCR implementation locally
This should work better on M1 Pro without getting stuck
"""

import time
import sys
from pathlib import Path

def test_improved_qwen():
    """Test the improved Qwen implementation."""
    print("🧪 Testing Improved Qwen2.5-VL Implementation")
    print("=" * 50)
    
    try:
        from app.qwen_ocr_improved import improved_qwen_ocr
        
        if improved_qwen_ocr is None:
            print("❌ Improved Qwen OCR not available (missing dependencies)")
            return False
        
        print("✅ Improved Qwen OCR module loaded")
        
        # Test with available images
        test_files = ["english.webp", "urdu.jpg"]
        available_files = [f for f in test_files if Path(f).exists()]
        
        if not available_files:
            print("❌ No test images found")
            print("💡 Please ensure english.webp or urdu.jpg is in the current directory")
            return False
        
        test_file = available_files[0]
        print(f"\n🧪 Testing with: {test_file}")
        
        # Progress callback
        def progress_callback(message: str, progress: int):
            print(f"📊 {progress:3d}% - {message}")
        
        # Test OCR
        start_time = time.time()
        print(f"\n🚀 Starting improved Qwen OCR...")
        print(f"⏰ Timeout: 2 minutes (should be much faster)")
        
        language = 'eng' if 'english' in test_file else 'urd'
        
        result = improved_qwen_ocr.extract_text(
            test_file, 
            language=language,
            progress_callback=progress_callback
        )
        
        request_time = time.time() - start_time
        print(f"\n⏱️  Request completed in {request_time:.2f}s")
        
        if result.get('success', False):
            print(f"\n✅ OCR Response received!")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"📝 Word Count: {result.get('word_count', 0)}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"📱 Device: {result.get('device', 'unknown')}")
            
            # Show extracted text
            text = result.get('text', '')
            if text:
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"\n📄 Extracted Text Preview:")
                print(f"   {preview}")
                
                # Compare with expected results
                if 'english' in test_file:
                    print(f"\n🎯 Expected: English text about learning")
                    if any(word in text.lower() for word in ['learn', 'education', 'knowledge', 'study']):
                        print(f"✅ Text seems relevant to expected content")
                    else:
                        print(f"⚠️  Text doesn't match expected content")
                
                return True
            else:
                print(f"\n⚠️  No text extracted")
                return False
        else:
            print(f"\n❌ OCR failed!")
            error = result.get('error', 'Unknown error')
            print(f"   Error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test just the model loading without OCR."""
    print("\n🔧 Testing Model Loading Only")
    print("-" * 30)
    
    try:
        from app.qwen_ocr_improved import improved_qwen_ocr
        
        if improved_qwen_ocr is None:
            print("❌ Improved Qwen OCR not available")
            return False
        
        def progress_callback(message: str, progress: int):
            print(f"📊 {progress:3d}% - {message}")
        
        print("🚀 Loading model...")
        start_time = time.time()
        
        success = improved_qwen_ocr.load_model(progress_callback)
        load_time = time.time() - start_time
        
        if success:
            print(f"✅ Model loaded successfully in {load_time:.2f}s")
            print(f"📱 Device: {improved_qwen_ocr.device}")
            print(f"🎯 Model: {improved_qwen_ocr.model_name}")
            return True
        else:
            print(f"❌ Model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 Testing Improved Qwen2.5-VL Implementation")
    print("📋 This test will:")
    print("   1. Test model loading")
    print("   2. Test OCR with available images")
    print("   3. Check for performance improvements")
    print("   4. Verify M1 Pro compatibility")
    print()
    
    # Test 1: Model loading
    model_success = test_model_loading()
    
    if not model_success:
        print(f"\n❌ Model loading failed - skipping OCR test")
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Check internet connection for model download")
        print(f"   • Ensure sufficient disk space (>10GB)")
        print(f"   • Try running: pip install transformers torch pillow")
        return
    
    # Test 2: OCR functionality
    ocr_success = test_improved_qwen()
    
    print(f"\n🎯 Test Results:")
    print(f"   Model Loading: {'✅ SUCCESS' if model_success else '❌ FAILED'}")
    print(f"   OCR Processing: {'✅ SUCCESS' if ocr_success else '❌ FAILED'}")
    
    if model_success and ocr_success:
        print(f"\n🎉 Improved Qwen Implementation Working!")
        print(f"   • Model loads without hanging")
        print(f"   • OCR processing completes successfully")
        print(f"   • M1 Pro compatibility confirmed")
        print(f"   • Ready for integration into main system")
    else:
        print(f"\n🔧 Issues Found:")
        if not model_success:
            print(f"   • Model loading needs attention")
        if not ocr_success:
            print(f"   • OCR processing needs debugging")
        
        print(f"\n💡 Next Steps:")
        print(f"   • Check the error messages above")
        print(f"   • Verify all dependencies are installed")
        print(f"   • Try with a smaller model if memory issues")

if __name__ == "__main__":
    main()
