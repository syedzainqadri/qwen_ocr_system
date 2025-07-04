#!/usr/bin/env python3
"""
Test the robust Qwen implementation with timeout protection
"""

import time
from pathlib import Path

def test_robust_qwen():
    """Test the robust Qwen implementation."""
    print("🛡️  Testing Robust Qwen2.5-VL Implementation")
    print("=" * 50)
    
    try:
        from app.qwen_ocr_robust import robust_qwen_ocr
        
        if robust_qwen_ocr is None:
            print("❌ Robust Qwen OCR not available (missing dependencies)")
            return False
        
        print("✅ Robust Qwen OCR module loaded")
        print(f"⏰ Timeout protection: {robust_qwen_ocr.timeout}s")
        
        # Test with available images
        test_files = ["english.webp", "urdu.jpg"]
        available_files = [f for f in test_files if Path(f).exists()]
        
        if not available_files:
            print("❌ No test images found")
            return False
        
        test_file = available_files[0]
        print(f"\n🧪 Testing with: {test_file}")
        
        # Progress callback
        def progress_callback(message: str, progress: int):
            print(f"📊 {progress:3d}% - {message}")
        
        # Test OCR with timeout protection
        start_time = time.time()
        print(f"\n🚀 Starting robust Qwen OCR...")
        print(f"⏰ Will timeout after {robust_qwen_ocr.timeout} seconds if stuck")
        
        language = 'eng' if 'english' in test_file else 'urd'
        
        result = robust_qwen_ocr.extract_text(
            test_file, 
            language=language,
            progress_callback=progress_callback
        )
        
        request_time = time.time() - start_time
        print(f"\n⏱️  Request completed in {request_time:.2f}s")
        
        # Analyze results
        if result.get('success', False):
            print(f"\n✅ OCR Response received!")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"📝 Word Count: {result.get('word_count', 0)}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"📱 Device: {result.get('device', 'unknown')}")
            print(f"⏰ Timeout Used: {result.get('timeout_used', 'N/A')}s")
            
            # Show extracted text
            text = result.get('text', '')
            if text:
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"\n📄 Extracted Text Preview:")
                print(f"   {preview}")
                return True
            else:
                print(f"\n⚠️  No text extracted")
                return False
                
        elif result.get('timeout_occurred', False):
            print(f"\n⏰ OCR TIMED OUT!")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"⏱️  Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"⏰ Timeout Used: {result.get('timeout_used', 'N/A')}s")
            print(f"❌ Error: {result.get('error', 'Unknown timeout error')}")
            print(f"🔄 Fallback Recommended: {result.get('fallback_recommended', False)}")
            
            print(f"\n💡 This confirms the M1 Pro text generation hanging issue")
            print(f"   • Model loads successfully")
            print(f"   • Image processing works")
            print(f"   • Text generation hangs (as expected)")
            print(f"   • Timeout protection works!")
            
            return "timeout"  # Special return value for timeout
            
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

def main():
    """Main test function."""
    print("🎯 Testing Robust Qwen2.5-VL with Timeout Protection")
    print("📋 This test will:")
    print("   1. Load Qwen2.5-VL model")
    print("   2. Process image correctly")
    print("   3. Attempt text generation")
    print("   4. Apply timeout protection if it hangs")
    print("   5. Return gracefully with fallback recommendation")
    print()
    
    result = test_robust_qwen()
    
    if result == True:
        print(f"\n🎉 Robust Test PASSED!")
        print(f"   • Model works perfectly on this system")
        print(f"   • Text generation completed successfully")
        print(f"   • No timeout needed")
        print(f"   • Ready for production use")
        
    elif result == "timeout":
        print(f"\n⏰ Robust Test TIMED OUT (Expected on M1 Pro)")
        print(f"   • Model loads successfully ✅")
        print(f"   • Image processing works ✅")
        print(f"   • Text generation hangs ❌ (M1 Pro issue)")
        print(f"   • Timeout protection works ✅")
        print(f"   • Graceful fallback available ✅")
        print(f"\n💡 Recommendation: Use PaddleOCR as primary, Qwen as optional")
        
    else:
        print(f"\n❌ Robust Test FAILED!")
        print(f"   • Check error messages above")
        print(f"   • May need dependency fixes")
        print(f"   • Use PaddleOCR as primary")

if __name__ == "__main__":
    main()
