from io import BytesIO
from random import choices, uniform
import json
from pathlib import Path
import base64

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

app = FastAPI(title="Oral Cancer Detection API (Vercel)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel-friendly demo labels.
CLASS_LABELS = {
    0: "Normal",
    1: "Cancerous",
}


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize(target_size, Image.Resampling.LANCZOS)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "vercel-demo",
        "model_loaded": False,
        "message": "TensorFlow model inference is disabled on Vercel serverless due package size limits.",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        preprocess_image(image)

        predicted_class = choices([0, 1], weights=[70, 30])[0]
        confidence = uniform(0.75, 0.98) if predicted_class == 0 else uniform(0.65, 0.95)

        result = {
            "prediction": CLASS_LABELS[predicted_class],
            "confidence": round(confidence * 100, 2),
            "risk_level": get_risk_level(confidence),
            "recommendations": get_recommendations(predicted_class, confidence),
            "demo_mode": True,
            "runtime": "vercel-python",
        }
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


def get_risk_level(confidence: float) -> str:
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.6:
        return "Moderate"
    return "Low"


def get_recommendations(predicted_class: int, confidence: float) -> list[str]:
    if predicted_class == 1:
        return [
            "Potentially suspicious pattern detected.",
            "Please consult a qualified oncologist or pathologist.",
            "Use clinical tests/biopsy for confirmation.",
            "This Vercel endpoint is running in demo mode.",
        ]

    if confidence < 0.7:
        return [
            "Appears normal with moderate confidence.",
            "Consider follow-up if symptoms persist.",
            "Maintain regular oral checkups.",
            "This Vercel endpoint is running in demo mode.",
        ]

    return [
        "No suspicious pattern detected.",
        "Maintain oral hygiene and regular screening.",
        "Consult a professional for any persistent symptoms.",
        "This Vercel endpoint is running in demo mode.",
    ]


@app.get("/api/confusion-matrix")
async def get_confusion_matrix():
    """Return confusion matrix visualization image"""
    try:
        cm_path = Path(__file__).parent.parent / "models" / "confusion_matrix.png"
        
        if not cm_path.exists():
            # Return demo confusion matrix data if image doesn't exist
            return JSONResponse(content={
                "status": "demo",
                "message": "Confusion matrix not generated yet. Run generate_confusion_matrix.py to generate it.",
                "demo_mode": True,
                "metrics": {
                    "sensitivity": 0.92,
                    "specificity": 0.88,
                    "accuracy": 0.90,
                    "f1_score": 0.89,
                    "roc_auc": 0.95,
                    "tn": 44,
                    "fp": 6,
                    "fn": 4,
                    "tp": 46
                }
            })
        
        return FileResponse(cm_path, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve confusion matrix: {e}")


@app.get("/api/metrics")
async def get_model_metrics():
    """Return detailed model metrics"""
    try:
        # Try to load metrics from a metrics file if it exists
        metrics_path = Path(__file__).parent.parent / "models" / "metrics.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                return JSONResponse(content=metrics)
        
        # Return demo metrics if file doesn't exist
        demo_metrics = {
            "status": "demo",
            "mode": "vercel-demo",
            "model_type": "DenseNet121",
            "dataset": "Histopathology Images",
            "performance": {
                "sensitivity": 0.92,  # TP / (TP + FN)
                "specificity": 0.88,  # TN / (TN + FP)
                "accuracy": 0.90,     # (TP + TN) / Total
                "precision": 0.88,    # TP / (TP + FP)
                "recall": 0.92,       # Same as sensitivity
                "f1_score": 0.89,
                "roc_auc": 0.95
            },
            "confusion_matrix": {
                "true_negatives": 44,
                "false_positives": 6,
                "false_negatives": 4,
                "true_positives": 46,
                "total_samples": 100
            },
            "class_names": ["Normal", "Cancerous"],
            "demo_mode": True,
            "note": "These are demo metrics. Run generate_confusion_matrix.py to generate real metrics."
        }
        return JSONResponse(content=demo_metrics)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {e}")


@app.get("/api/confusion-matrix/base64")
async def get_confusion_matrix_base64():
    """Return confusion matrix as base64-encoded image"""
    try:
        cm_path = Path(__file__).parent.parent / "models" / "confusion_matrix.png"
        
        if not cm_path.exists():
            return JSONResponse(content={
                "status": "not_found",
                "message": "Confusion matrix not generated yet",
                "demo_mode": True
            })
        
        with open(cm_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        return JSONResponse(content={
            "status": "success",
            "image": f"data:image/png;base64,{image_data}",
            "format": "base64"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode image: {e}")
