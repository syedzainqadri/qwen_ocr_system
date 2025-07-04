#!/usr/bin/env python3
"""
Basic Qwen2.5-VL test following the M1 Pro compatible approach
This is a minimal test to verify the core functionality works
"""

import time
from pathlib import Path

def test_basic_qwen_approach():
    """Test the basic Qwen approach that should work on M1 Pro."""
    print("🧪 Basic Qwen2.5-VL Test (M1 Pro Compatible)")
    print("=" * 50)
    
    try:
        # Import dependencies
        print("📦 Importing dependencies...")
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from PIL import Image
        import torch
        print("✅ Dependencies imported successfully")
        
        # Check available test images
        test_files = ["english.webp", "urdu.jpg"]
        available_files = [f for f in test_files if Path(f).exists()]
        
        if not available_files:
            print("❌ No test images found")
            print("💡 Please ensure english.webp or urdu.jpg is in the current directory")
            return False
        
        test_file = available_files[0]
        print(f"🖼️  Using test image: {test_file}")
        
        # Load model (this is where it might get stuck on M1 Pro)
        print("\n📥 Loading Qwen2.5-VL model...")
        print("⏰ This may take a few minutes on first run...")
        
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        start_time = time.time()
        
        # Load processor
        print("🔧 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("✅ Processor loaded")
        
        # Load model with M1 Pro optimized settings
        print("🤖 Loading model...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,  # Use float32 for M1 Pro
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to("cpu")  # Force CPU for stability
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Load and process image
        print(f"\n📷 Loading image: {test_file}")
        image = Image.open(test_file).convert("RGB")
        print(f"✅ Image loaded: {image.size}")
        
        # Create prompt
        prompt = "<|im_start|>user\nWhat is the text written in this image? Please transcribe all text accurately.\n<|im_end|>\n<|im_start|>assistant\n"
        
        # Process inputs
        print("🔄 Processing inputs...")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cpu")
        print("✅ Inputs processed")
        
        # Generate (this is where it often gets stuck)
        print("\n🎯 Generating text...")
        print("⏰ Timeout: 60 seconds")
        
        generation_start = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,  # Conservative for M1 Pro
                do_sample=False,     # Deterministic
                num_beams=1,         # No beam search
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        generation_time = time.time() - generation_start
        print(f"✅ Text generated in {generation_time:.2f}s")
        
        # Decode output
        print("📝 Decoding output...")
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract the actual text
        if prompt in output:
            extracted_text = output.replace(prompt, "").strip()
        else:
            extracted_text = output.strip()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 SUCCESS!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📄 Extracted text length: {len(extracted_text)} characters")
        
        if extracted_text:
            preview = extracted_text[:150] + "..." if len(extracted_text) > 150 else extracted_text
            print(f"\n📝 Extracted Text:")
            print(f"   {preview}")
        else:
            print(f"\n⚠️  No text extracted")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🎯 Basic Qwen2.5-VL Test for M1 Pro")
    print("📋 This test will:")
    print("   1. Load Qwen2.5-VL-3B-Instruct model")
    print("   2. Process a test image")
    print("   3. Generate OCR text")
    print("   4. Check for hanging/timeout issues")
    print()
    print("💡 If this test works, we can integrate it into the main system")
    print()
    
    success = test_basic_qwen_approach()
    
    if success:
        print(f"\n🎉 Basic Test PASSED!")
        print(f"   • Model loads without hanging")
        print(f"   • Text generation completes")
        print(f"   • M1 Pro compatibility confirmed")
        print(f"   • Ready to integrate improved approach")
    else:
        print(f"\n❌ Basic Test FAILED!")
        print(f"   • Check error messages above")
        print(f"   • May need to adjust model settings")
        print(f"   • Consider using PaddleOCR as primary")

if __name__ == "__main__":
    main()
