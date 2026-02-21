from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
from pathlib import Path
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Oral Cancer Detection API - DEMO MODE")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels
CLASS_LABELS = {
    0: "Normal",
    1: "Cancerous"
}

DEMO_MODE = True  # Set to False when you have a trained model

@app.on_event("startup")
async def load_model():
    """Check for trained model on startup"""
    logger.warning("⚠️  RUNNING IN DEMO MODE")
    logger.warning("⚠️  No model loaded - predictions are simulated")
    logger.warning("⚠️  To use real AI:")
    logger.warning("    1. Install TensorFlow: pip install tensorflow>=2.20.0 numpy>=2.0.0 opencv-python>=4.10.0 matplotlib>=3.9.0")
    logger.warning("    2. Train your model: python train_model.py")
    logger.warning("    3. Use app.py instead of app_demo.py")

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Preprocess image for display"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    return image

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        content = f.read()
        # Add demo mode banner
        content = content.replace(
            '<body class="bg-gray-50">',
            '''<body class="bg-gray-50">
            <div style="background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%); color: white; padding: 12px; text-align: center; font-weight: bold;">
                ⚠️ DEMO MODE - AI predictions are simulated. Install TensorFlow and train a model for real predictions.
            </div>'''
        )
        return HTMLResponse(content=content)

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    DEMO: Simulate oral cancer prediction
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess (just for validation)
        processed_image = preprocess_image(image)
        
        # DEMO MODE: Generate random predictions
        logger.info("🎲 Generating simulated prediction (DEMO MODE)")
        
        # 70% chance of normal, 30% chance of cancerous (more realistic)
        predicted_class = random.choices([0, 1], weights=[70, 30])[0]
        
        # Generate realistic confidence (higher for normal, varied for cancerous)
        if predicted_class == 0:
            confidence = random.uniform(0.75, 0.98)
        else:
            confidence = random.uniform(0.65, 0.95)
        
        result = {
            "prediction": CLASS_LABELS[predicted_class],
            "confidence": round(confidence * 100, 2),
            "risk_level": get_risk_level(confidence),
            "recommendations": get_recommendations(predicted_class, confidence),
            "demo_mode": True
        }
        
        logger.info(f"Demo Prediction: {result['prediction']} ({result['confidence']}%)")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def get_risk_level(confidence: float) -> str:
    """Determine risk level based on confidence"""
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Moderate"
    else:
        return "Low"

def get_recommendations(predicted_class: int, confidence: float) -> list:
    """Provide recommendations based on prediction"""
    if predicted_class == 1:  # Cancerous
        return [
            "⚠️ Urgent: Please consult an oncologist immediately",
            "📋 Get a professional biopsy for confirmation",
            "🏥 Schedule an appointment at a cancer care center",
            "📱 Do not delay medical consultation",
            "🎭 NOTE: This is a DEMO prediction, not real AI analysis"
        ]
    else:  # Normal
        if confidence < 0.7:
            return [
                "✅ Appears normal, but confidence is moderate",
                "🔍 Consider a follow-up check if symptoms persist",
                "👨‍⚕️ Maintain regular dental check-ups",
                "🦷 Practice good oral hygiene",
                "🎭 NOTE: This is a DEMO prediction, not real AI analysis"
            ]
        else:
            return [
                "✅ No signs of cancer detected",
                "🦷 Maintain regular oral hygiene",
                "👨‍⚕️ Continue routine dental check-ups",
                "🥗 Follow a healthy lifestyle",
                "🎭 NOTE: This is a DEMO prediction, not real AI analysis"
            ]

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "demo",
        "model_loaded": False,
        "message": "Running in demo mode. Install TensorFlow and train model for real predictions."
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # Create necessary directories
    Path("static").mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("🏥 Oral Cancer Detection - DEMO MODE")
    print("="*60)
    print("⚠️  This is a DEMO version with simulated predictions")
    print("\n📋 To use real AI predictions:")
    print("   1. Install TensorFlow: pip install -r requirements.txt")
    print("   2. Prepare your dataset in dataset/train/ and dataset/validation/")
    print("   3. Train the model: python train_model.py")
    print("   4. Run the real app: python app.py")
    print("\n🌐 Starting demo server...")
    print("   Open: http://localhost:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
