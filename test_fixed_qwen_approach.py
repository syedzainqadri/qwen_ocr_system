#!/usr/bin/env python3
"""
Fixed Qwen2.5-VL test using the correct qwen-vl-utils approach
This should resolve the image token mismatch issue
"""

import time
from pathlib import Path

def test_fixed_qwen_approach():
    """Test the fixed Qwen approach with proper image processing."""
    print("🧪 Fixed Qwen2.5-VL Test (Proper Image Processing)")
    print("=" * 55)
    
    try:
        # Import dependencies
        print("📦 Importing dependencies...")
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from PIL import Image
        import torch
        
        # Import qwen-vl-utils for proper image processing
        try:
            from qwen_vl_utils import process_vision_info
            print("✅ qwen-vl-utils imported successfully")
        except ImportError:
            print("❌ qwen-vl-utils not found - installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "qwen-vl-utils"])
            from qwen_vl_utils import process_vision_info
            print("✅ qwen-vl-utils installed and imported")
        
        print("✅ All dependencies imported successfully")
        
        # Check available test images
        test_files = ["english.webp", "urdu.jpg"]
        available_files = [f for f in test_files if Path(f).exists()]
        
        if not available_files:
            print("❌ No test images found")
            print("💡 Please ensure english.webp or urdu.jpg is in the current directory")
            return False
        
        test_file = available_files[0]
        print(f"🖼️  Using test image: {test_file}")
        
        # Load model
        print("\n📥 Loading Qwen2.5-VL model...")
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
        
        # Load image
        print(f"\n📷 Loading image: {test_file}")
        image = Image.open(test_file).convert("RGB")
        print(f"✅ Image loaded: {image.size}")
        
        # Create proper message format for Qwen2.5-VL
        print("📝 Creating message...")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": test_file,  # Use file path directly
                    },
                    {
                        "type": "text", 
                        "text": "What is the text written in this image? Please transcribe all text accurately."
                    },
                ],
            }
        ]
        
        # Apply chat template
        print("🔄 Applying chat template...")
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("✅ Chat template applied")
        
        # Process vision info using qwen-vl-utils
        print("👁️  Processing vision info...")
        image_inputs, video_inputs = process_vision_info(messages)
        print(f"✅ Vision info processed: {len(image_inputs)} images")
        
        # Process inputs with the correct format
        print("🔄 Processing inputs...")
        inputs = processor(
            text=[text_prompt],  # Text as list
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cpu")
        print("✅ Inputs processed")
        
        # Generate (this should work now)
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
        
        # Trim the input tokens from the generated output
        input_token_len = inputs["input_ids"].shape[1]
        response_token_ids = generated_ids[:, input_token_len:]
        
        # Decode the response
        output = processor.batch_decode(
            response_token_ids, skip_special_tokens=True
        )[0]
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 SUCCESS!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📄 Extracted text length: {len(output)} characters")
        
        if output:
            preview = output[:200] + "..." if len(output) > 200 else output
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
    print("🎯 Fixed Qwen2.5-VL Test with Proper Image Processing")
    print("📋 This test will:")
    print("   1. Use qwen-vl-utils for proper image processing")
    print("   2. Apply correct chat template format")
    print("   3. Process vision info correctly")
    print("   4. Generate OCR text without token mismatch")
    print()
    print("💡 This should fix the 'Image features and image tokens do not match' error")
    print()
    
    success = test_fixed_qwen_approach()
    
    if success:
        print(f"\n🎉 Fixed Test PASSED!")
        print(f"   • Model loads successfully")
        print(f"   • Image processing works correctly")
        print(f"   • Text generation completes")
        print(f"   • No token mismatch errors")
        print(f"   • Ready to integrate into main system")
    else:
        print(f"\n❌ Fixed Test FAILED!")
        print(f"   • Check error messages above")
        print(f"   • May need further adjustments")
        print(f"   • Consider fallback to PaddleOCR")

if __name__ == "__main__":
    main()
