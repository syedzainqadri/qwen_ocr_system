#!/usr/bin/env python3
"""
Test script for cloud-deployed OCR system.
"""

import requests
import time
import sys
from pathlib import Path

def test_cloud_deployment(base_url):
    """Test the cloud-deployed OCR system."""
    print(f"🌐 Testing Cloud Deployment: {base_url}")
    print("=" * 60)
    
    # Remove trailing slash
    base_url = base_url.rstrip('/')
    
    # Test 1: Health Check
    print("1. 🏥 Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Status: {health.get('status', 'unknown')}")
            print(f"   📱 Device: {health.get('device', 'unknown')}")
            print(f"   🤖 Model Loaded: {health.get('model_loaded', False)}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Web Interface
    print("\n2. 🌐 Web Interface...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("   ✅ Web interface accessible")
            if "Qwen OCR System" in response.text:
                print("   ✅ Correct page loaded")
            else:
                print("   ⚠️  Page loaded but content unexpected")
        else:
            print(f"   ❌ Web interface failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Web interface error: {e}")
    
    # Test 3: API Documentation
    print("\n3. 📚 API Documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("   ✅ API docs accessible")
        else:
            print(f"   ❌ API docs failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ API docs error: {e}")
    
    # Test 4: OCR API (if test image available)
    print("\n4. 🔍 OCR API Test...")
    test_files = ["english.webp", "urdu.jpg", "test.jpg", "test.png"]
    test_file = None
    
    for file in test_files:
        if Path(file).exists():
            test_file = file
            break
    
    if test_file:
        print(f"   📄 Testing with: {test_file}")
        try:
            with open(test_file, 'rb') as f:
                files = {'file': (test_file, f, 'image/jpeg')}
                data = {'language': 'eng', 'model': 'paddle'}
                
                start_time = time.time()
                response = requests.post(f"{base_url}/ocr", files=files, data=data, timeout=60)
                request_time = time.time() - start_time
                
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ OCR API working!")
                print(f"   🔧 Engine: {result.get('engine', 'unknown')}")
                print(f"   ⏱️  Response Time: {request_time:.2f}s")
                print(f"   📊 Confidence: {result.get('confidence', 0):.1f}%")
                print(f"   📝 Word Count: {result.get('word_count', 0)}")
                
                text = result.get('text', '')
                if text:
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"   📄 Text Preview: {preview}")
                else:
                    print(f"   ⚠️  No text extracted")
            else:
                print(f"   ❌ OCR API failed: {response.status_code}")
                try:
                    error = response.json()
                    print(f"   📋 Error: {error}")
                except:
                    print(f"   📋 Raw response: {response.text[:200]}")
        except Exception as e:
            print(f"   ❌ OCR API error: {e}")
    else:
        print("   ⚠️  No test images found (english.webp, urdu.jpg, test.jpg, test.png)")
        print("   💡 Upload a test image to fully test the OCR API")
    
    # Test 5: Performance Check
    print("\n5. ⚡ Performance Check...")
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/health", timeout=10)
        response_time = time.time() - start_time
        
        if response_time < 1.0:
            print(f"   ✅ Fast response: {response_time:.3f}s")
        elif response_time < 3.0:
            print(f"   ⚠️  Moderate response: {response_time:.3f}s")
        else:
            print(f"   ❌ Slow response: {response_time:.3f}s")
    except Exception as e:
        print(f"   ❌ Performance check error: {e}")
    
    print("\n🎯 Cloud Deployment Test Complete!")
    print("=" * 60)
    
    return True

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_cloud_deployment.py <URL>")
        print("Example: python test_cloud_deployment.py https://your-app.railway.app")
        sys.exit(1)
    
    base_url = sys.argv[1]
    
    print("🧪 Cloud OCR System Test")
    print("📋 This will test:")
    print("   • Health endpoint")
    print("   • Web interface")
    print("   • API documentation")
    print("   • OCR functionality")
    print("   • Performance")
    print()
    
    success = test_cloud_deployment(base_url)
    
    if success:
        print("\n🎉 Your cloud deployment is working!")
        print("🌐 Access your OCR system at:")
        print(f"   {base_url}")
        print("\n📚 API Documentation:")
        print(f"   {base_url}/docs")
        print("\n🔗 Share this URL with your users!")
    else:
        print("\n❌ Some tests failed. Check the deployment logs.")
        print("💡 Common issues:")
        print("   • Server still starting up (wait a few minutes)")
        print("   • Environment variables not set correctly")
        print("   • Insufficient memory/CPU resources")

if __name__ == "__main__":
    main()
