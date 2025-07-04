# 🚀 Coolify Deployment Guide for Qwen OCR System

## 📋 Pre-Deployment Checklist

✅ **Repository Ready**: All code committed and ready for GitHub  
✅ **Docker Support**: Dockerfile and docker-compose.yml configured  
✅ **Environment Variables**: Production settings defined  
✅ **Health Checks**: `/health` endpoint implemented  
✅ **Model Toggle**: Users can choose between Qwen2.5-VL and PaddleOCR  
✅ **Progress Tracking**: Real-time updates during OCR processing  
✅ **Training Infrastructure**: PaddleOCR training system included  

## 🌐 Step 1: Push to GitHub

1. **Create GitHub Repository**:
   - Go to https://github.com/new
   - Repository name: `qwen-ocr-system`
   - Description: `Powerful OCR system with Qwen2.5-VL-3B and PaddleOCR`
   - Make it **Public** (required for Coolify free tier)
   - **Don't** initialize with README (we already have one)

2. **Push Code**:
   ```bash
   ./push_to_github.sh https://github.com/yourusername/qwen-ocr-system.git
   ```

## 🚀 Step 2: Deploy to Coolify

### 2.1 Create New Project

1. Open your Coolify dashboard
2. Click **"New Project"**
3. Choose **"Git Repository"**
4. Connect your GitHub account (if not already connected)
5. Select the `qwen-ocr-system` repository

### 2.2 Configure Deployment

**Build Settings**:
- **Build Pack**: Docker
- **Dockerfile Path**: `Dockerfile` (default)
- **Build Context**: `.` (root directory)

**Environment Variables**:
```bash
ENVIRONMENT=production
PYTHONUNBUFFERED=1
TRANSFORMERS_CACHE=/app/.cache
HF_HOME=/app/.cache
```
(Note: Don't set PORT - let Coolify assign it automatically)

**Resource Allocation**:
- **Memory**: 4GB (minimum for AI models)
- **CPU**: 2 cores (recommended)
- **Storage**: 10GB (for model cache)

### 2.3 Configure Volumes (Optional but Recommended)

For persistent model cache and uploads:

```yaml
Volumes:
  - model_cache:/app/.cache
  - uploads:/app/uploads
  - training_data:/app/training_data
```

### 2.4 Health Check Configuration

Coolify will automatically use the health check defined in the Dockerfile:
- **Endpoint**: `/health`
- **Interval**: 30s
- **Timeout**: 10s
- **Retries**: 3

## 🧪 Step 3: Test Deployment

After deployment completes:

1. **Check Health**:
   ```bash
   curl https://your-app.coolify.io/health
   ```

2. **Test Web Interface**:
   Visit: `https://your-app.coolify.io`

3. **Test API**:
   ```bash
   curl -X POST https://your-app.coolify.io/ocr \
     -F "file=@test-image.jpg" \
     -F "language=eng" \
     -F "model=paddle"
   ```

4. **Run Full Test Suite**:
   ```bash
   python test_cloud_deployment.py https://your-app.coolify.io
   ```

## ⚡ Expected Performance on Coolify

### PaddleOCR (Recommended for Production)
- **Speed**: 2-5 seconds
- **Accuracy**: Good, trainable
- **Memory Usage**: ~1-2GB
- **Best for**: Production workloads

### Qwen2.5-VL-3B
- **Speed**: 30-60 seconds (cloud hardware)
- **Accuracy**: Excellent
- **Memory Usage**: ~3-4GB
- **Best for**: High-accuracy requirements

## 🔧 Troubleshooting

### Common Issues:

1. **Build Timeout**:
   - Increase build timeout in Coolify settings
   - Models download during first build (can take 10-15 minutes)

2. **Memory Issues**:
   - Increase memory allocation to 6-8GB
   - Use PaddleOCR only mode for lower memory usage

3. **Slow Model Loading**:
   - Enable persistent volumes for model cache
   - Models will be cached after first run

4. **Health Check Failures**:
   - Models need time to load on first startup
   - Increase health check start period to 120s

### Debug Commands:

```bash
# Check logs
coolify logs your-app-name

# Check resource usage
coolify stats your-app-name

# Restart deployment
coolify restart your-app-name
```

## 🎯 Production Optimization

### For Better Performance:

1. **Enable Model Caching**:
   - Use persistent volumes
   - Models download once, cached forever

2. **Resource Scaling**:
   - Start with 4GB RAM, scale up if needed
   - Monitor memory usage in Coolify dashboard

3. **Load Balancing**:
   - Deploy multiple instances for high traffic
   - Use Coolify's built-in load balancing

### For Cost Optimization:

1. **Use PaddleOCR Mode**:
   - Set default model to "paddle"
   - Much lower resource requirements

2. **Auto-scaling**:
   - Configure auto-scaling based on CPU/memory
   - Scale down during low usage

## 🌟 Features Available After Deployment

✅ **Web Interface**: Clean UI for uploading images and getting OCR results  
✅ **Model Toggle**: Users can choose between Qwen2.5-VL and PaddleOCR  
✅ **Multi-language**: Support for English, Urdu, Arabic, French, German, Spanish  
✅ **Progress Tracking**: Real-time updates during processing  
✅ **REST API**: Complete API with Swagger documentation at `/docs`  
✅ **Training Ready**: PaddleOCR can be trained with custom data  
✅ **Production Ready**: Health checks, logging, error handling  

## 📊 Monitoring

### Key Metrics to Monitor:

- **Response Time**: Should be <5s for PaddleOCR, <60s for Qwen
- **Memory Usage**: Should stay under allocated limits
- **Error Rate**: Monitor failed OCR requests
- **Model Loading Time**: First request may be slower

### Coolify Dashboard:

- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, network usage
- **Health**: Service health status
- **Deployments**: Deployment history and rollbacks

## 🎉 Success!

Once deployed, your OCR system will be:

🌐 **Publicly Accessible**: Available at your Coolify URL  
⚡ **High Performance**: Running on cloud infrastructure  
🔄 **Auto-scaling**: Handles traffic spikes automatically  
🛡️ **Secure**: HTTPS enabled by default  
📊 **Monitored**: Full observability through Coolify  
🚀 **Production Ready**: Suitable for real-world usage  

**Your OCR system is now ready to handle production workloads! 🎯**

---

## 📞 Support

- **Coolify Documentation**: https://coolify.io/docs
- **GitHub Issues**: Report issues in your repository
- **Testing**: Use `test_cloud_deployment.py` for validation
