# 🚀 Render Deployment Guide for VisionMate

## 📋 Prerequisites

1. GitHub account with your VisionMate repository
2. Render account (free tier works!) - [Sign up here](https://render.com)
3. All code pushed to your GitHub repository

## 🎯 Quick Deploy Steps

### Step 1: Connect Repository

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml`

### Step 2: Configure (Auto-detected from render.yaml)

The `render.yaml` file already contains all configuration:
- ✅ Python 3.9
- ✅ System packages (ffmpeg, libsm6, libxext6)
- ✅ Streamlit configuration
- ✅ Health check endpoint
- ✅ Environment variables

### Step 3: Deploy

1. Click **"Apply"**
2. Wait 5-10 minutes for first deployment
   - Installing system packages: ~2 min
   - Installing Python packages: ~3 min
   - Starting Streamlit: ~1 min
   - Model loading: ~2-3 min

### Step 4: Access Your App

Once deployed, your app will be available at:
```
https://visionmate.onrender.com
```
(or the custom URL Render assigns)

## 🔧 Configuration Details

### System Packages (render.yaml)
```yaml
buildCommand: >-
  apt-get update -qq &&
  apt-get install -y -qq ffmpeg libsm6 libxext6 &&
  pip install --upgrade pip &&
  pip install -r requirements.txt
```

**Why these packages?**
- `ffmpeg` - Video codec for WebRTC
- `libsm6` - X11 Session Management for OpenCV
- `libxext6` - X11 extensions for OpenCV

### Streamlit Configuration
```yaml
startCommand: >-
  streamlit run app.py
  --server.port $PORT
  --server.address 0.0.0.0
  --server.headless true
  --server.enableCORS false
  --server.enableXsrfProtection false
```

### Health Check
```yaml
healthCheckPath: /_stcore/health
```
This tells Render where to check if the app is running.

## 🌐 Render vs Streamlit Cloud

| Feature | Render | Streamlit Cloud |
|---------|--------|-----------------|
| **Free Tier** | 750 hours/month | Unlimited |
| **Startup Time** | 5-10 min | 2-3 min |
| **Custom Domain** | ✅ Yes | ✅ Yes |
| **Auto-deploy** | ✅ Yes | ✅ Yes |
| **Sleep on inactivity** | ✅ Yes (15 min) | ❌ No |
| **WebRTC Support** | ✅ Excellent | ✅ Excellent |
| **Memory** | 512 MB (free) | 1 GB |
| **Build Control** | ✅ Full | ⚠️ Limited |

## ⚙️ Environment Variables (Optional)

### Add TURN Server (for restrictive networks)
In Render Dashboard → Your Service → Environment:
```
TURN_URL=turn:your-server.com:3478
TURN_USER=username
TURN_CREDENTIAL=password
```

**Note**: Not required! Multiple STUN servers work for 99% of users.

## 🐛 Troubleshooting

### Issue 1: Deployment Fails During Build
**Error**: `E: Unable to correct problems, you have held broken packages`

**Fix**: Render.yaml already updated with compatible packages. If issue persists:
1. Check Render build logs
2. Verify `render.yaml` is in repository root
3. Try changing region to `oregon` or `frankfurt`

### Issue 2: App Starts but Shows Error
**Check**: Render logs (Dashboard → Your Service → Logs)

**Common causes**:
- Model files not committed to Git
- Files > 100MB (use Git LFS)
- Import errors

**Solution**: Check logs for specific error message

### Issue 3: "Service Unavailable" 
**Cause**: Render free tier sleeps after 15 minutes of inactivity

**Solution**: 
- First request will take 30-60 seconds (cold start)
- Upgrade to paid tier for 24/7 uptime
- Or keep app alive with uptime monitoring (e.g., UptimeRobot)

### Issue 4: WebRTC Camera Not Working
**Causes**:
1. HTTP instead of HTTPS → Render provides HTTPS automatically ✅
2. Browser permissions → User must allow camera
3. Corporate firewall → Add TURN server credentials

**Solution**: Render URLs are HTTPS by default, so camera should work!

## 📊 Expected Performance

### Free Tier Limits
- **Memory**: 512 MB RAM
- **CPU**: Shared
- **Bandwidth**: 100 GB/month
- **Build minutes**: 500 min/month

### Typical Usage
- **Model loading**: ~2-3 minutes (first request)
- **Inference latency**: 50-100ms
- **Concurrent users**: 5-10 (free tier)
- **Cold start**: 30-60 seconds

## 🔄 Continuous Deployment

Once set up, every push to main branch triggers auto-deploy:
```bash
git add .
git commit -m "Update feature"
git push origin main
```

Render will:
1. Pull latest code
2. Run build command
3. Start Streamlit
4. Health check passes
5. Route traffic to new deployment

## 📁 Required Files in Repository

Make sure these exist:
```
✅ render.yaml          # Deployment config (root)
✅ requirements.txt     # Python dependencies
✅ packages.txt         # Not used on Render (apt packages in render.yaml)
✅ app.py              # Main application
✅ models/             # Model files (.h5)
✅ .gitignore          # Exclude visionmate.db, __pycache__, etc.
```

## 🎯 Post-Deployment Checklist

After successful deployment:

- [ ] App loads without errors
- [ ] Check logs: `Dashboard → Service → Logs`
- [ ] Test camera permission prompt
- [ ] Face detection works
- [ ] Model predictions display
- [ ] Analytics tab shows data
- [ ] No console errors in browser DevTools (F12)

## 🔒 Security Considerations

**On Render**:
- ✅ HTTPS enabled by default
- ✅ Environment variables encrypted
- ✅ SQLite database persisted (in free tier: resets on redeploy)
- ⚠️ For production: Consider PostgreSQL addon

**To add persistent database**:
1. Add Render PostgreSQL database
2. Update `database/db_manager.py` to use PostgreSQL
3. Add `psycopg2-binary` to requirements.txt

## 💰 Cost Considerations

### Free Tier
- **Perfect for**: Testing, demos, FYP projects
- **Limitations**: Sleeps after 15 min, 512 MB RAM
- **Best for**: 10-50 daily active users

### Starter Plan ($7/month)
- **Benefits**: No sleep, 512 MB RAM, faster builds
- **Best for**: Production with <100 users

### Standard Plan ($25/month)
- **Benefits**: 2 GB RAM, priority support
- **Best for**: Production with 100+ users

## 🚀 Advanced: Custom Domain

1. Go to Dashboard → Your Service → Settings
2. Add Custom Domain
3. Update DNS records (A record or CNAME)
4. Render provides free SSL certificate

## 📝 Quick Reference

### View Logs
```bash
# In Render Dashboard:
Your Service → Logs → Select time range
```

### Restart Service
```bash
# In Render Dashboard:
Your Service → Manual Deploy → Deploy latest commit
```

### Check Health
```bash
curl https://visionmate.onrender.com/_stcore/health
```

## 🎓 Summary

**Render is excellent for VisionMate because**:
- ✅ Full control over system packages
- ✅ Better for WebRTC (no firewall issues)
- ✅ Easy deployment with render.yaml
- ✅ Free tier sufficient for FYP project
- ✅ Professional deployment option

**Deploy now**: Just connect your repo and click Apply! 🚀

---

**Need Help?**
- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- Check `/app/STREAMLIT_DEPLOYMENT_FIX.md` for common issues

**Status**: ✅ Ready for Render deployment with updated render.yaml
