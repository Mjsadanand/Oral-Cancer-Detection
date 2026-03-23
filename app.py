from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
MODEL_PATH = "models/oral_cancer_model.h5"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup: Load the model
    global model
    try:
        if Path(MODEL_PATH).exists():
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✅ Model loaded successfully!")
        else:
            logger.warning(f"⚠️ Model not found at {MODEL_PATH}. Please train and save your model.")
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
    
    yield
    
    # Shutdown cleanup (if needed)
    pass

app = FastAPI(title="Oral Cancer Detection API - Histopathology Analysis", lifespan=lifespan)

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

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Preprocess histopathology image for model prediction"""
    # Convert to RGB if necessary (important for histopathology images)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image with high-quality resampling
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array with proper dtype
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0,1] - must match training preprocessing
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict oral cancer from uploaded histopathology image
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists and is trained."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Log image info
        logger.info(f"Processing image: {file.filename}, Size: {image.size}, Mode: {image.mode}")
        
        # Preprocess
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get prediction results (binary classification)
        raw_score = float(predictions[0][0])
        
        # Determine class with threshold
        threshold = 0.5
        predicted_class = 1 if raw_score > threshold else 0
        confidence = raw_score if predicted_class == 1 else 1 - raw_score
        
        result = {
            "prediction": CLASS_LABELS.get(predicted_class, "Unknown"),
            "confidence": round(confidence * 100, 2),
            "raw_score": round(raw_score, 4),
            "risk_level": get_risk_level(predicted_class, confidence),
            "recommendations": get_recommendations(predicted_class, confidence)
        }
        
        logger.info(f"Prediction: {result['prediction']} ({result['confidence']}%) - Risk: {result['risk_level']}")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def get_risk_level(predicted_class: int, confidence: float) -> str:
    """Determine risk level based on prediction and confidence for histopathology"""
    if predicted_class == 1:  # Cancerous
        if confidence >= 0.85:
            return "Very High"
        elif confidence >= 0.70:
            return "High"
        else:
            return "Moderate"
    else:  # Normal
        if confidence >= 0.85:
            return "Very Low"
        elif confidence >= 0.70:
            return "Low"
        else:
            return "Uncertain"

def get_recommendations(predicted_class: int, confidence: float) -> list:
    """Provide recommendations based on histopathology prediction"""
    if predicted_class == 1:  # Cancerous
        if confidence >= 0.85:
            return [
                "⚠️ URGENT: High confidence of malignancy detected",
                "🏭 Immediate referral to oncology specialist required",
                "📋 Professional histopathological confirmation needed",
                "🏥 Discuss treatment options with oncologist immediately",
                "📞 Schedule biopsy and imaging studies"
            ]
        elif confidence >= 0.70:
            return [
                "⚠️ Warning: Suspicious findings detected in tissue sample",
                "📋 Get professional histopathological review",
                "🏭 Consult with an oncologist or pathologist",
                "🔬 Additional diagnostic tests recommended",
                "📱 Do not delay medical consultation"
            ]
        else:
            return [
                "⚠️ Possible abnormality detected (moderate confidence)",
                "🔍 Further examination recommended",
                "👨‍⚕️ Consult with a specialist",
                "📋 Consider additional biopsies from different areas",
                "📊 Follow-up testing advised"
            ]
    else:  # Normal
        if confidence >= 0.85:
            return [
                "✅ No malignancy detected with high confidence",
                "🧬 Tissue appears normal in this sample",
                "👨‍⚕️ Continue routine medical check-ups",
                "🥗 Maintain healthy oral hygiene practices",
                "📅 Regular screening as per medical guidelines"
            ]
        elif confidence >= 0.70:
            return [
                "✅ Appears normal, but moderate confidence",
                "🔍 Consider follow-up if clinical symptoms persist",
                "👨‍⚕️ Consult with physician about findings",
                "🥗 Maintain good oral health",
                "📋 May need additional sampling if indicated"
            ]
        else:
            return [
                "⚠️ Uncertain classification - low confidence",
                "🔍 Additional testing strongly recommended",
                "👨‍⚕️ Professional pathologist review needed",
                "📋 Consider repeat biopsy or additional samples",
                "📞 Schedule follow-up examination"
            ]

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("uploads").mkdir(exist_ok=True)
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
