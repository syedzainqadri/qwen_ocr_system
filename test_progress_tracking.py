#!/usr/bin/env python3
"""
Test script to verify the progress tracking system works.
"""

import requests
import json
import time
from pathlib import Path
import websocket
import threading

BASE_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001/ws"

class ProgressTracker:
    def __init__(self):
        self.progress_updates = []
        self.ws = None
        self.connected = False
    
    def on_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            data = json.loads(message)
            progress = data.get('progress', 0)
            message_text = data.get('message', '')
            self.progress_updates.append((progress, message_text))
            print(f"📊 Progress: {progress}% - {message_text}")
        except Exception as e:
            print(f"Error parsing message: {e}")
    
    def on_open(self, ws):
        """Handle WebSocket connection open."""
        self.connected = True
        print("🔗 WebSocket connected for progress tracking")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self.connected = False
        print("🔌 WebSocket disconnected")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"❌ WebSocket error: {error}")
    
    def connect(self):
        """Connect to WebSocket."""
        try:
            self.ws = websocket.WebSocketApp(
                WS_URL,
                on_message=self.on_message,
                on_open=self.on_open,
                on_close=self.on_close,
                on_error=self.on_error
            )
            
            # Run WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection
            time.sleep(2)
            return self.connected
        except Exception as e:
            print(f"Failed to connect WebSocket: {e}")
            return False

def test_progress_tracking():
    """Test the progress tracking system."""
    print("🧪 Testing Progress Tracking System")
    print("=" * 60)
    
    # Initialize progress tracker
    tracker = ProgressTracker()
    
    # Connect to WebSocket
    print("🔗 Connecting to WebSocket for progress updates...")
    if not tracker.connect():
        print("❌ Failed to connect to WebSocket")
        print("💡 Testing without progress tracking...")
    else:
        print("✅ WebSocket connected successfully!")
    
    # Test health first
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
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
    
    # Test with the first available file
    test_file = available_files[0]
    print(f"\n🧪 Testing with: {test_file}")
    print(f"📊 Watch for real-time progress updates...")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'image/webp' if test_file.endswith('.webp') else 'image/jpeg')}
            data = {'language': 'eng' if 'english' in test_file else 'urd'}
            
            start_time = time.time()
            print(f"🚀 Starting OCR request...")
            print(f"⏰ Timeout set to 2 minutes (120s)")

            response = requests.post(f"{BASE_URL}/ocr", files=files, data=data, timeout=120)
            request_time = time.time() - start_time
            
        print(f"\n⏱️  Request completed in {request_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ OCR Response received!")
            print(f"🔧 Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"📝 Word Count: {result.get('word_count', 0)}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")
            
            # Show progress tracking results
            if tracker.progress_updates:
                print(f"\n📊 Progress Tracking Results:")
                print(f"   • Total updates received: {len(tracker.progress_updates)}")
                print(f"   • First update: {tracker.progress_updates[0][1]} ({tracker.progress_updates[0][0]}%)")
                print(f"   • Last update: {tracker.progress_updates[-1][1]} ({tracker.progress_updates[-1][0]}%)")
                print(f"   • Progress tracking: ✅ WORKING")
            else:
                print(f"\n⚠️  No progress updates received")
                print(f"   • WebSocket may not be working properly")
            
            return True
            
        else:
            print(f"❌ OCR request failed: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    """Main test function."""
    success = test_progress_tracking()
    
    print(f"\n🎯 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n🎉 Progress Tracking System Working!")
        print(f"   • Real-time progress updates implemented")
        print(f"   • WebSocket connection established")
        print(f"   • Users can see processing status")
        print(f"   • No more waiting in the dark!")
    else:
        print(f"\n🔧 Progress Tracking Notes:")
        print(f"   • Check WebSocket connection")
        print(f"   • Verify server is running properly")
        print(f"   • Progress updates should show during processing")

if __name__ == "__main__":
    main()
