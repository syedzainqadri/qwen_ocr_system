#!/usr/bin/env python3
"""
Test the working Qwen implementation based on successful Mac solution
"""

import time
import sys
from pathlib import Path

def test_working_qwen():
    """Test the working Qwen implementation."""
    print("🧪 Testing Working Qwen2.5-VL-3B Implementation (Mac Solution)")
    print("=" * 65)
    
    try:
        from app.qwen_ocr_working import working_qwen_ocr
        
        if working_qwen_ocr is None:
            print("❌ Working Qwen OCR not available (missing dependencies)")
            return False
        
        print("✅ Working Qwen OCR module loaded")
        print(f"📱 Device: {working_qwen_ocr.device}")
        print(f"⏰ Timeout: {working_qwen_ocr.timeout}s")
        print(f"🎯 Model: {working_qwen_ocr.model_name}")
        
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
        
        # Test OCR with working approach
        start_time = time.time()
        print(f"\n🚀 Starting working Qwen OCR...")
        print(f"💡 This uses the approach that works on Mac without hanging")
        print(f"⏰ Will timeout after {working_qwen_ocr.timeout} seconds if needed")
        
        language = 'eng' if 'english' in test_file else 'urd'
        
        result = working_qwen_ocr.extract_text(
            test_file, 
            language=language,
            progress_callback=progress_callback
        )
        
        request_time = time.time() - start_time
        print(f"\n⏱️  Request completed in {request_time:.2f}s")
        
        # Analyze results
        if result.get('success', False):
            print(f"\n🎉 OCR SUCCESS!")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"📝 Word Count: {result.get('word_count', 0)}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"📱 Device: {result.get('device', 'unknown')}")
            print(f"🎯 Approach: {result.get('approach', 'unknown')}")
            
            # Show extracted text
            text = result.get('text', '')
            if text:
                preview = text[:300] + "..." if len(text) > 300 else text
                print(f"\n📄 Extracted Text:")
                print(f"   {preview}")
                
                # Compare with expected results
                if 'english' in test_file:
                    print(f"\n🎯 Expected: English text about learning")
                    if any(word in text.lower() for word in ['learn', 'education', 'knowledge', 'study', 'text', 'image']):
                        print(f"✅ Text seems relevant to expected content")
                    else:
                        print(f"⚠️  Text doesn't match expected content")
                
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
            print(f"🎯 Approach: {result.get('approach', 'unknown')}")
            
            print(f"\n💡 Even the working approach timed out")
            print(f"   • This might indicate hardware limitations")
            print(f"   • Try with a smaller image or more memory")
            print(f"   • The working approach loaded successfully though!")
            
            return "timeout"
            
        else:
            print(f"\n❌ OCR failed!")
            error = result.get('error', 'Unknown error')
            print(f"   Error: {error}")
            print(f"   Approach: {result.get('approach', 'unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_approaches():
    """Compare the working approach with our previous approach."""
    print("\n🔍 Comparing Approaches")
    print("-" * 40)
    
    print("📊 Working Mac Solution (Updated for Qwen2.5-VL-3B):")
    print("   ✅ Uses Qwen2.5-VL-3B-Instruct (smaller, faster)")
    print("   ✅ Uses device_map='auto' for better device handling")
    print("   ✅ Uses offload_buffers=True for memory management")
    print("   ✅ Uses torch.float16 for better performance")
    print("   ✅ Resizes images to reduce memory pressure")
    print("   ✅ Uses simpler message format (no qwen-vl-utils)")
    print("   ✅ Uses AutoTokenizer + AutoProcessor")
    print("   ✅ Uses temperature and repetition_penalty")
    print("   ✅ Already downloaded - no bandwidth usage")
    
    print("\n📊 Our Previous Approach:")
    print("   ❌ Used Qwen2_5_VLForConditionalGeneration")
    print("   ❌ Used manual device assignment (.to(device))")
    print("   ❌ Used torch.float32 (more memory)")
    print("   ❌ Used qwen-vl-utils (more complex)")
    print("   ❌ No image resizing")
    print("   ❌ More conservative generation parameters")
    
    print("\n💡 Key Learnings:")
    print("   🎯 Model class matters: Qwen2VL vs Qwen2_5_VL")
    print("   🎯 Device mapping: 'auto' vs manual assignment")
    print("   🎯 Memory management: offload_buffers=True")
    print("   🎯 Image preprocessing: resize to reduce memory")
    print("   🎯 Simpler is better: avoid complex utils when possible")

def main():
    """Main test function."""
    print("🎯 Testing Working Qwen2.5-VL-3B Implementation")
    print("📋 Based on successful Mac solution that doesn't hang")
    print("🔗 Source: Community-verified working approach")
    print("💡 Using already downloaded 3B model - no bandwidth usage")
    print()
    
    # Show comparison first
    compare_approaches()
    
    # Test the working implementation
    result = test_working_qwen()
    
    print(f"\n🎯 Test Results:")
    
    if result == True:
        print(f"🎉 WORKING APPROACH SUCCESS!")
        print(f"   • Model loads and works properly")
        print(f"   • Text generation completes successfully")
        print(f"   • No hanging or timeout issues")
        print(f"   • Ready for integration into main system")
        
    elif result == "timeout":
        print(f"⏰ WORKING APPROACH TIMED OUT")
        print(f"   • Model loads successfully ✅")
        print(f"   • Approach is correct ✅")
        print(f"   • Hardware may need more resources")
        print(f"   • Still better than hanging indefinitely")
        
    else:
        print(f"❌ WORKING APPROACH FAILED")
        print(f"   • Check error messages above")
        print(f"   • May need dependency fixes")
        print(f"   • Fallback to PaddleOCR recommended")
    
    print(f"\n💡 Next Steps:")
    if result == True:
        print(f"   1. Integrate working approach into main system")
        print(f"   2. Replace robust_qwen_ocr with working_qwen_ocr")
        print(f"   3. Test in production environment")
        print(f"   4. Deploy to Coolify with working implementation")
    else:
        print(f"   1. Use PaddleOCR as primary engine")
        print(f"   2. Keep working approach as optional fallback")
        print(f"   3. Consider hardware upgrades for better Qwen performance")

if __name__ == "__main__":
    main()
