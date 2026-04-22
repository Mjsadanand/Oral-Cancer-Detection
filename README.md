# 🏥 Oral Cancer Detection System

An AI-powered web application for oral cancer detection using deep learning and a modern web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🤖 **AI-Powered Detection**: Deep learning model for accurate oral cancer detection
- ⚡ **Fast Analysis**: Get results in under 2 seconds
- 🎨 **Modern UI**: Beautiful, Dribbble-inspired interface
- 📊 **Detailed Reports**: Comprehensive analysis with confidence scores
- 🔒 **Privacy First**: Images processed locally, not stored
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU (optional, for faster training)

### Installation

1. **Clone or navigate to project directory**
```bash
cd "V:\A\Soumya"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset** (see Dataset Structure below)

4. **Train the model**
```bash
python train_model.py
```

5. **Run the application**
```bash
python app.py
```

6. **Open your browser**
```
http://localhost:8000
```

## 📁 Dataset Structure

Organize your dataset as follows:

```
dataset/
├── train/
│   ├── normal/
│   │   ├── normal_001.jpg
│   │   ├── normal_002.jpg
│   │   └── ...
│   └── cancerous/
│       ├── cancer_001.jpg
│       ├── cancer_002.jpg
│       └── ...
└── validation/
    ├── normal/
    │   ├── normal_val_001.jpg
    │   └── ...
    └── cancerous/
        ├── cancer_val_001.jpg
        └── ...
```

**Dataset Tips:**
- Use at least 500+ images per class for good results
- Split: 80% training, 20% validation
- Image format: JPG, PNG
- Recommended size: 224x224 or higher
- Balance your dataset (similar number of normal vs cancerous images)

## 🎯 Model Training

### Training Options

The `train_model.py` script supports multiple architectures:

1. **EfficientNetB0** (Recommended - Best accuracy)
2. **ResNet50** (Good for larger datasets)
3. **MobileNetV2** (Fastest, mobile-friendly)
4. **Custom CNN** (Build from scratch)

To change the model, edit `train_model.py`:

```python
model = create_model(model_type='efficientnet')  # or 'resnet50', 'mobilenet', 'custom'
```

### Training Configuration

Adjust these parameters in `train_model.py`:

```python
IMG_SIZE = (224, 224)      # Image size
BATCH_SIZE = 32            # Batch size (reduce if out of memory)
EPOCHS = 50                # Maximum epochs
LEARNING_RATE = 0.001      # Learning rate
```

### Expected Training Time

- **With GPU**: 10-30 minutes (depending on dataset size)
- **Without GPU**: 1-3 hours

### Model Performance

After training, you'll see:
- ✅ Validation Accuracy
- ✅ Precision & Recall
- ✅ F1 Score
- 📈 Training history plots

## 🌐 Running the Web App

### Start the Server

```bash
python app.py
```

The server will start at `http://localhost:8000`

### API Endpoints

#### `GET /`
Serves the main web interface

#### `POST /api/predict`
Predicts oral cancer from uploaded image

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "prediction": "Normal",
  "confidence": 95.5,
  "risk_level": "Low",
  "recommendations": [
    "✅ No signs of cancer detected",
    "🦷 Maintain regular oral hygiene",
    "..."
  ]
}
```

#### `GET /api/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🎨 UI Features

The web interface includes:

- **Drag & Drop Upload**: Easy image upload
- **Real-time Preview**: See your image before analysis
- **Progress Indicator**: Visual feedback during analysis
- **Detailed Results**: 
  - Prediction (Normal/Cancerous)
  - Confidence percentage
  - Risk level assessment
  - Personalized recommendations
- **Responsive Design**: Works on all devices
- **Smooth Animations**: Modern, polished UI

## 🔧 Customization

### Change Model Path

Edit `app.py`:
```python
MODEL_PATH = "models/your_model.h5"
```

### Modify Class Labels

Edit `app.py`:
```python
CLASS_LABELS = {
    0: "Normal",
    1: "Cancerous",
    # Add more classes if needed
}
```

### Adjust Image Preprocessing

Edit the `preprocess_image` function in `app.py`:
```python
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    # Your custom preprocessing
    pass
```

### Customize UI Colors

Edit `static/index.html` gradient classes:
```html
<style>
    .gradient-bg {
        background: linear-gradient(135deg, #YOUR_COLOR_1, #YOUR_COLOR_2);
    }
</style>
```

## 📊 Model Evaluation

To evaluate your model on test data:

```python
from tensorflow.keras.models import load_model

model = load_model('models/oral_cancer_model.h5')

# Create test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Evaluate
results = model.evaluate(test_generator)
print(f"Test Accuracy: {results[1] * 100:.2f}%")
```

## 🐛 Troubleshooting

### Model Not Loading
- Ensure the model file exists at `models/oral_cancer_model.h5`
- Check file permissions
- Verify TensorFlow installation

### Out of Memory During Training
- Reduce `BATCH_SIZE` in `train_model.py`
- Use a smaller model (MobileNetV2)
- Reduce image size

### Slow Predictions
- Use a GPU for faster inference
- Convert model to TensorFlow Lite
- Use a lighter architecture (MobileNetV2)

### Poor Accuracy
- Increase dataset size
- Ensure balanced classes
- Try data augmentation
- Use transfer learning (pre-trained models)
- Train for more epochs

## 📈 Improving Model Performance

1. **More Data**: Collect more training images
2. **Data Augmentation**: Already included in training script
3. **Transfer Learning**: Use pre-trained models (EfficientNet, ResNet)
4. **Hyperparameter Tuning**: Adjust learning rate, batch size
5. **Ensemble Methods**: Combine multiple models
6. **Cross-validation**: Validate on multiple splits

## 🔒 Security & Privacy

⚠️ **Important Medical AI Considerations:**

- This is a **screening tool**, not a diagnostic tool
- Always consult medical professionals for diagnosis
- Do not use as sole basis for medical decisions
- Images are processed locally and not stored
- HIPAA compliance required for production use

## 📝 Project Structure

```
V:\A\Soumya/
├── app.py                    # FastAPI backend
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── static/
│   └── index.html           # Frontend UI
├── models/
│   ├── oral_cancer_model.h5 # Trained model
│   └── training_history.png # Training plots
├── dataset/                  # Training data (you create this)
│   ├── train/
│   │   ├── normal/
│   │   └── cancerous/
│   └── validation/
│       ├── normal/
│       └── cancerous/
└── uploads/                  # Temporary upload folder
```

## 🚀 Deployment

### Local Deployment
Already configured! Just run:
```bash
python app.py
```

### Cloud Deployment Options

#### 0. **Render (Full Stack - Simplest)**

Deploy frontend + backend together on Render with zero additional setup.

- Frontend: static site (free)
- Backend: Python web service with TensorFlow (~$7/month)
- Automatic proxy routing between them

See `RENDER_DEPLOYMENT.md` for step-by-step.

Quick start:

```bash
git push origin main
# Then on Render: New → Blueprint, select your repo
# Auto-deployed from render.yaml
```

#### 1. **Vercel (Recommended for UI only) + Separate Backend**

Vercel serverless Python functions have strict package size limits. TensorFlow-based inference from `app.py` is too large for Vercel Lambda storage.

This repository includes a Vercel-ready setup:

- `api/index.py` -> lightweight serverless API (demo mode)
- `api/requirements.txt` -> minimal dependencies for Vercel build
- `vercel.json` -> routing for `/api/*` and static UI
- `.vercelignore` -> excludes large training/model assets

Deploy steps:

```bash
vercel
vercel --prod
```

To enable real TensorFlow inference, deploy `app.py` on Render, Azure, Railway, AWS, or GCP, then add to Vercel env vars:

```bash
INFERENCE_API_BASE_URL=https://your-real-backend.example.com
```

The Vercel function in `api/index.py` supports two modes:

- **Demo mode** (default): simulated prediction response.
- **Proxy mode**: forwards `/api/predict` and `/api/health` to your real backend.

See `AZURE_BACKEND_DEPLOY.md` or `RENDER_DEPLOYMENT.md` for backend deployment.

#### 1. **Google Cloud Platform**
```bash
gcloud app deploy
```

#### 2. **AWS EC2**
```bash
# Install dependencies on EC2
# Run with gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

#### 2.1 **Azure App Service (TensorFlow Backend via Docker)**

Use the ready deployment files in this repo:

- `Dockerfile.azure`
- `requirements-inference.txt`
- `AZURE_BACKEND_DEPLOY.md`

Quick flow:

1. Build and push container to Azure Container Registry.
2. Deploy Linux Web App from that container image.
3. Set `WEBSITES_PORT=8000`.
4. Put Azure backend URL into Vercel env var:

```bash
INFERENCE_API_BASE_URL=https://<your-app-name>.azurewebsites.net
```

5. Redeploy Vercel.

Detailed step-by-step commands are in `AZURE_BACKEND_DEPLOY.md`.

**Tip**: For simpler single-service deployment, use Render instead (see option 0 above).

#### 3. **Docker**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4. **Heroku**
```bash
heroku create
git push heroku main
```

## 🎓 Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Image Classification Guide](https://www.tensorflow.org/tutorials/images/classification)
- [Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

**MEDICAL DISCLAIMER**: This software is for educational and research purposes only. It is NOT intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding medical conditions.

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Email: your-email@example.com

## 🙏 Acknowledgments

- TensorFlow team for the amazing ML framework
- FastAPI for the modern web framework
- The medical AI research community
- All contributors and supporters

---

**Built with ❤️ for better healthcare through AI**

**Version**: 1.0.0  
**Last Updated**: February 2026
