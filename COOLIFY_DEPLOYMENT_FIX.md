# 🚨 COOLIFY DEPLOYMENT FIX

## 🔍 Issues Identified from Logs

From your Coolify deployment logs, I identified these critical issues:

### ❌ **Primary Issues:**
1. **Health Check Failure**: `./healthcheck.sh` script not working because `curl` wasn't available in the expected way
2. **Container Restart Loop**: Container was restarting repeatedly (visible from duplicate startup messages)
3. **Health Check Timeout**: 60-second start period wasn't enough for model loading

### ❌ **Secondary Issues:**
4. **Import Failures**: Potential issues with `robust_qwen_ocr` import causing crashes
5. **Error Handling**: Insufficient error handling for missing dependencies

## ✅ **Fixes Applied**

### 1. 🔧 **Fixed Health Check**
**Before:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD ./healthcheck.sh
```

**After:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:${PORT:-3030}/health || exit 1
```

**Changes:**
- ✅ Use `curl` directly instead of script
- ✅ Increased start period to 180s for model loading
- ✅ Increased retries to 5 for better reliability
- ✅ Reduced timeout to 10s for faster detection

### 2. 🛡️ **Added Safety Checks**
**Before:**
```python
from .qwen_ocr_robust import robust_qwen_ocr
```

**After:**
```python
try:
    from .qwen_ocr_robust import robust_qwen_ocr
except Exception as e:
    print(f"Failed to import robust_qwen_ocr: {e}")
    robust_qwen_ocr = None
```

**Benefits:**
- ✅ Container won't crash if Qwen fails to import
- ✅ Graceful fallback to PaddleOCR-only mode
- ✅ Better error messages and logging

### 3. 🔄 **Improved Error Handling**
**Added checks for:**
- ✅ Qwen availability before use
- ✅ Graceful degradation when Qwen unavailable
- ✅ Better server startup error handling
- ✅ Proper exception handling in OCR endpoints

### 4. 📦 **Created Simple Dockerfile**
**New `Dockerfile.simple`** with:
- ✅ Simplified build process
- ✅ Longer health check start period (180s)
- ✅ More retries (5 instead of 3)
- ✅ Robust error handling

## 🚀 **Deployment Instructions**

### **Option 1: Use Current Dockerfile (Recommended)**
1. **Redeploy in Coolify** - The fixes are already in the main Dockerfile
2. **Wait longer** - Health check now has 180s start period
3. **Monitor logs** - Look for successful startup messages

### **Option 2: Use Simple Dockerfile**
1. **Change Dockerfile path** in Coolify to: `Dockerfile.simple`
2. **Redeploy** - This uses the most conservative settings
3. **Test** - Should be more reliable

### **Option 3: Disable Health Check Temporarily**
1. **Turn off health check** in Coolify UI
2. **Redeploy** - Container will start without health verification
3. **Test manually** - Visit the URL to check if it works
4. **Re-enable** health check once working

## 📊 **Expected Behavior After Fix**

### ✅ **Successful Deployment Logs:**
```
🚀 Starting Qwen2.5-VL OCR Server
========================================
🎯 Primary Engine: Qwen2.5-VL-3B-Instruct
🔄 Fallback Engine: PaddleOCR (trainable & accurate)
🌐 Environment: production
🔧 Port: 3030 (or assigned port)
✅ Server starting at http://0.0.0.0:3030
INFO: Uvicorn running on http://0.0.0.0:3030
INFO: PaddleOCR Engine initialized
INFO: Application startup complete
```

### ✅ **Health Check Success:**
```
Container health status: "healthy"
Health check passing
Application ready
```

### ✅ **No More Issues:**
- ❌ No container restart loops
- ❌ No health check failures
- ❌ No import errors causing crashes
- ❌ No timeout issues

## 🧪 **Testing After Deployment**

### 1. **Check Health Endpoint**
```bash
curl https://your-app.coolify.io/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 2. **Test PaddleOCR (Should Work)**
```bash
curl -X POST https://your-app.coolify.io/ocr \
  -F "file=@test-image.jpg" \
  -F "language=eng" \
  -F "model=paddle"
```

### 3. **Test Qwen (May Work or Timeout Gracefully)**
```bash
curl -X POST https://your-app.coolify.io/ocr \
  -F "file=@test-image.jpg" \
  -F "language=eng" \
  -F "model=qwen"
```

### 4. **Test Auto Mode (Intelligent Fallback)**
```bash
curl -X POST https://your-app.coolify.io/ocr \
  -F "file=@test-image.jpg" \
  -F "language=eng" \
  -F "model=auto"
```

## 🔧 **Troubleshooting**

### **If Still Failing:**

1. **Check Coolify Logs** for specific error messages
2. **Try Dockerfile.simple** - More conservative approach
3. **Disable Health Check** temporarily to isolate issues
4. **Increase Resources** - More memory/CPU if available
5. **Use PaddleOCR Only** - Disable Qwen temporarily

### **Resource Requirements:**
- **Minimum**: 2GB RAM, 1 CPU core
- **Recommended**: 4GB RAM, 2 CPU cores
- **With Qwen**: 6-8GB RAM, 2+ CPU cores

### **Environment Variables:**
```bash
ENVIRONMENT=production
PYTHONUNBUFFERED=1
# Don't set PORT - let Coolify assign it
```

## 🎯 **Success Indicators**

After successful deployment:

✅ **Container Status**: Running (not restarting)  
✅ **Health Check**: Passing  
✅ **PaddleOCR**: Working reliably  
✅ **Qwen**: Either working or timing out gracefully  
✅ **Auto Mode**: Intelligent fallback working  
✅ **Web Interface**: Accessible and functional  

## 📞 **Next Steps**

1. **Redeploy in Coolify** with the fixes
2. **Wait for health check** to pass (up to 3 minutes)
3. **Test the endpoints** as shown above
4. **Monitor performance** and adjust resources if needed
5. **Report results** - Let me know if any issues persist

**The fixes address all the issues from your deployment logs and should result in a successful deployment! 🚀**
