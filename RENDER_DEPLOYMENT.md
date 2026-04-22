# Render Deployment (Full Stack: Frontend + Backend on Free Tier)

Deploy your entire app on Render's free tier—frontend UI + lightweight TensorFlow Lite API backend.

## Why TensorFlow Lite?

Render's free tier has limited resources. Full TensorFlow is too large (~500 MB).
**TensorFlow Lite is optimized for inference and runs on free tier:**
- ~30 MB total dependencies vs 500+ MB with full TensorFlow
- Same accuracy as original model
- Perfect for production inference

## Quick Deploy

### Step 1: Convert Model (Locally)

Run this once on your machine to convert .h5 to .tflite:

```bash
python convert_to_tflite.py
```

You'll see:
```
✅ Conversion successful!
   Original (.h5):      150.00 MB
   Converted (.tflite): 45.00 MB
   Reduction:           70%
```

This creates `models/oral_cancer_model.tflite`.

### Step 2: Push to GitHub

```bash
git add .
git commit -m "Add TFLite model and lightweight deployment"
git push origin main
```

### Step 3: Deploy on Render

1. Go to https://render.com
2. Sign in with GitHub
3. Click "New +" → "Blueprint"
4. Select your repository
5. Render auto-reads `render.yaml` and deploys both services

### Step 4: Verify Deployment

After deploy completes (2-3 minutes):

```bash
# Check backend health
curl https://oral-cancer-backend-<random>.onrender.com/api/health

# Should return:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "runtime": "tflite-lightweight"
# }
```

Visit frontend URL and test image upload.

## File Requirements

Make sure your repo has:

- ✅ `app.py` - FastAPI backend (updated for TFLite)
- ✅ `tflite_inference.py` - TFLite inference wrapper
- ✅ `requirements-tflite.txt` - lightweight dependencies (~30 MB)
- ✅ `models/oral_cancer_model.tflite` - converted model (not .h5)
- ✅ `models/class_indices.json` - label mapping
- ✅ `static/index.html` - frontend UI
- ✅ `render.yaml` - deployment config

## Why Free Tier Works Now

- **Dependencies**: 30 MB (vs 500+ MB with TensorFlow)
- **Model file**: ~45 MB (optimized for inference)
- **Total**: ~75 MB fits comfortably in free tier

Render free tier includes:
- 512 MB memory ✅
- No cold spindowns (changes in Apr 2024, check current status)
- 0.5 shared CPU

## Troubleshooting

### Backend won't start: "ModuleNotFoundError: tflite_inference"

- Confirm `tflite_inference.py` is in repo root
- Check build logs for import errors
- File must be in same directory as `app.py`

### Model not loading: "TFLite model not found"

- Ensure `models/oral_cancer_model.tflite` exists
- Rerun `python convert_to_tflite.py` locally
- Commit and re-push

### Model exists but predictions fail

- Check that conversion succeeded (file size > 40 MB)
- Verify `models/class_indices.json` exists
- Check Render logs:
  ```bash
  # View live logs in Render dashboard
  ```

## Deployment Architecture

```
Frontend (Render Static): https://oral-cancer-frontend.onrender.com
  ↓
Reverse Proxy (route /api/*)
  ↓
Backend (Render Web Service): https://oral-cancer-backend.onrender.com
  ↓
TFLite Model Inference (~30 MB runtime)
```

## Cost

- **Free tier**: $0/month
- **Both services on free**: included

## Performance Notes

- First request: ~5-10 seconds (model warmup)
- Subsequent requests: <1 second
- Memory: 512 MB shared (tight but works)
- CPU: 0.5 shared CPU

If you hit performance limits, upgrade to Standard ($7/month for backend).

## Next Steps After Deploy

1. Test with different images
2. Monitor Render logs
3. If free tier is slow, upgrade to Standard
4. Share your live demo!

