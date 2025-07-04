#!/usr/bin/env python3
"""
Test script for Qwen OCR system to compare results with expected outputs.
"""

import requests
import json
import time
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8001"
TEST_IMAGES = [
    {
        "file": "english.webp",
        "language": "eng",
        "expected_keywords": ["text", "document", "english"],  # Add expected keywords
        "description": "English text document"
    },
    {
        "file": "urdu.jpg", 
        "language": "urd",
        "expected_keywords": ["اردو", "متن"],  # Add expected Urdu keywords
        "description": "Urdu text document"
    }
]

def test_health():
    """Test if the server is healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server Status: {data['status']}")
            print(f"📱 Model Loaded: {data['model_loaded']}")
            print(f"💻 Device: {data['device']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_ocr_image(image_info):
    """Test OCR on a single image."""
    print(f"\n🔍 Testing {image_info['description']} ({image_info['file']})")
    print("-" * 60)
    
    file_path = Path(image_info['file'])
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        # Prepare the request
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'image/webp' if file_path.suffix == '.webp' else 'image/jpeg')}
            data = {'language': image_info['language']}
            
            print(f"📤 Uploading {file_path.name} for {image_info['language']} OCR...")
            start_time = time.time()
            
            response = requests.post(f"{BASE_URL}/ocr", files=files, data=data, timeout=300)
            
            processing_time = time.time() - start_time
            
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ OCR Completed in {processing_time:.2f}s")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"📝 Word Count: {result.get('word_count', 0)}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"🖥️  Model: {result.get('model_name', 'unknown')}")
            print(f"💾 Device: {result.get('device', 'unknown')}")
            
            # Display extracted text
            text = result.get('text', '')
            if text:
                print(f"\n📄 Extracted Text:")
                print("=" * 40)
                print(text[:500] + ("..." if len(text) > 500 else ""))
                print("=" * 40)
                
                # Check for expected keywords
                found_keywords = []
                for keyword in image_info.get('expected_keywords', []):
                    if keyword.lower() in text.lower():
                        found_keywords.append(keyword)
                
                if found_keywords:
                    print(f"✅ Found expected keywords: {', '.join(found_keywords)}")
                else:
                    print(f"⚠️  No expected keywords found in text")
            else:
                print("❌ No text extracted")
            
            # Check for errors
            if result.get('error'):
                print(f"⚠️  Error reported: {result['error']}")
            
            return result
            
        else:
            print(f"❌ OCR failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"❌ Error during OCR: {e}")
        return None

def compare_results(results):
    """Compare results between different images and engines."""
    print(f"\n📊 Results Summary")
    print("=" * 60)
    
    for i, result in enumerate(results):
        if result:
            image_info = TEST_IMAGES[i]
            print(f"📁 {image_info['description']}:")
            print(f"   Engine: {result.get('engine', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.1f}%")
            print(f"   Words: {result.get('word_count', 0)}")
            print(f"   Time: {result.get('processing_time', 0):.2f}s")
            print(f"   Success: {'✅' if result.get('success', False) else '❌'}")
        else:
            print(f"📁 {TEST_IMAGES[i]['description']}: ❌ Failed")
        print()

def main():
    """Main test function."""
    print("🚀 Qwen OCR System Test Suite")
    print("=" * 60)
    
    # Test server health
    if not test_health():
        print("❌ Server not available. Please start the server first.")
        return
    
    print(f"\n🧪 Testing {len(TEST_IMAGES)} images...")
    
    # Test each image
    results = []
    for image_info in TEST_IMAGES:
        result = test_ocr_image(image_info)
        results.append(result)
        
        # Wait a bit between requests
        time.sleep(2)
    
    # Compare results
    compare_results(results)
    
    print("\n🎯 Test completed!")
    print("💡 Tip: You can also test via the web interface at http://localhost:8001")

if __name__ == "__main__":
    main()
