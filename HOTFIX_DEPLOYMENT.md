# 🚨 HOTFIX: Coolify Deployment Issues

## 🔍 Issues Identified

From the Coolify logs, we identified these problems:

1. **❌ NumPy Version Conflict**: 
   - Error: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.1"
   - **Fix**: Pin NumPy to version <2.0.0

2. **❌ Port Mismatch**:
   - Coolify assigns dynamic ports (like 3000), but health checks used fixed port 8001
   - **Fix**: Make port handling flexible and update health checks

3. **❌ Environment Detection**:
   - App detected "development" instead of "production"
   - **Fix**: Better environment detection logic

## 🛠️ Fixes Applied

### 1. NumPy Compatibility Fix

**Updated requirements.txt**:
```txt
numpy<2.0.0,>=1.24.0  # Fixed version constraint
```

**Updated Dockerfile**:
```dockerfile
# Install NumPy first to avoid conflicts
RUN pip install --no-cache-dir "numpy<2.0.0,>=1.24.0" && \
    pip install --no-cache-dir -r requirements.txt
```

### 2. Port Handling Fix

**Updated run_server.py**:
```python
# Accept any port from environment (default to 3030 instead of 8001)
port = int(os.getenv("PORT", 3030))

# Better environment detection
environment = os.getenv("ENVIRONMENT", "production")  # Default to production
if any(os.getenv(var) for var in ["RAILWAY_ENVIRONMENT", "RENDER", "HEROKU_APP_NAME", "COOLIFY_APP_ID"]):
    environment = "production"
```

**Updated Health Checks**:
```bash
# Flexible health check script
./healthcheck.sh  # Uses PORT environment variable
```

### 3. Production Dockerfile

Created `Dockerfile.production` with:
- Explicit NumPy version control
- Step-by-step dependency installation
- Better error handling
- Longer health check start period

## 🚀 Redeployment Steps

### Option 1: Quick Fix (Recommended)

1. **Commit the fixes**:
   ```bash
   git add .
   git commit -m "Fix NumPy compatibility and port handling for Coolify deployment"
   git push origin main
   ```

2. **Redeploy in Coolify**:
   - Go to your Coolify project
   - Click "Redeploy"
   - Wait for build to complete

### Option 2: Use Production Dockerfile

1. **Update Coolify settings**:
   - Change Dockerfile path to: `Dockerfile.production`
   - Redeploy

### Option 3: Environment Variables Fix

Set these in Coolify:
```bash
ENVIRONMENT=production
PYTHONUNBUFFERED=1
# Don't set PORT - let Coolify assign it dynamically
```

## 🧪 Testing After Fix

1. **Check logs** for successful startup:
   ```
   ✅ Server starting at http://0.0.0.0:3000  # (or whatever port Coolify assigns)
   ✅ INFO: Uvicorn running on http://0.0.0.0:3000
   ```

2. **Test health endpoint**:
   ```bash
   curl https://your-app.coolify.io/health
   ```

3. **Test web interface**:
   Visit: `https://your-app.coolify.io`

## 🔧 Expected Behavior After Fix

**Startup logs should show**:
```
🚀 Starting Qwen2.5-VL OCR Server
🌐 Environment: production
🔧 Port: 3000 (or whatever port Coolify assigns)
🔄 Reload: False
✅ Server starting at http://0.0.0.0:3000
INFO: Uvicorn running on http://0.0.0.0:3000
INFO: PaddleOCR Engine initialized
INFO: Qwen OCR Engine initialized
```

**No more errors about**:
- ❌ NumPy version conflicts
- ❌ Port mismatches
- ❌ Development mode in production

## 🚨 If Issues Persist

### Check Coolify Resource Allocation

**Minimum requirements**:
- **Memory**: 4GB (for AI models)
- **CPU**: 2 cores
- **Storage**: 10GB

### Alternative: PaddleOCR-Only Mode

If Qwen models are causing memory issues, temporarily disable them:

**In app/main.py**, change:
```python
# Force PaddleOCR only for initial deployment
FORCE_PADDLEOCR_ONLY = True

if FORCE_PADDLEOCR_ONLY or model == "paddle":
    # Use PaddleOCR only
```

### Debug Commands

**Check container status**:
```bash
# In Coolify terminal
docker ps
docker logs container-name
```

**Check resource usage**:
```bash
docker stats container-name
```

## 🎯 Success Indicators

After successful deployment:

✅ **Container starts without errors**  
✅ **Health check passes**  
✅ **Web interface loads**  
✅ **API responds to requests**  
✅ **OCR processing works**  

## 📞 Next Steps

1. **Commit and push fixes**
2. **Redeploy in Coolify**
3. **Test the deployment**
4. **Monitor resource usage**
5. **Scale up if needed**

**The fixes should resolve the NumPy conflict and port issues! 🚀**
