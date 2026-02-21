from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="Oral Cancer Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_PATH = "models/oral_cancer_model.h5"

# Class labels
CLASS_LABELS = {
    0: "Normal",
    1: "Cancerous"
}

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        if Path(MODEL_PATH).exists():
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✅ Model loaded successfully!")
        else:
            logger.warning(f"⚠️ Model not found at {MODEL_PATH}. Please train and save your model.")
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize to [0,1]
    
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
    Predict oral cancer from uploaded image
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
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
        
        # Preprocess
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get prediction results
        confidence = float(predictions[0][0])
        
        # Determine class (adjust based on your model output)
        if len(predictions[0]) > 1:
            # Multi-class classification
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class])
        else:
            # Binary classification
            predicted_class = 1 if confidence > 0.5 else 0
            confidence = confidence if predicted_class == 1 else 1 - confidence
        
        result = {
            "prediction": CLASS_LABELS.get(predicted_class, "Unknown"),
            "confidence": round(confidence * 100, 2),
            "risk_level": get_risk_level(confidence),
            "recommendations": get_recommendations(predicted_class, confidence)
        }
        
        logger.info(f"Prediction: {result['prediction']} ({result['confidence']}%)")
        
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
            "📱 Do not delay medical consultation"
        ]
    else:  # Normal
        if confidence < 0.7:
            return [
                "✅ Appears normal, but confidence is moderate",
                "🔍 Consider a follow-up check if symptoms persist",
                "👨‍⚕️ Maintain regular dental check-ups",
                "🦷 Practice good oral hygiene"
            ]
        else:
            return [
                "✅ No signs of cancer detected",
                "🦷 Maintain regular oral hygiene",
                "👨‍⚕️ Continue routine dental check-ups",
                "🥗 Follow a healthy lifestyle"
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
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
