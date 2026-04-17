from io import BytesIO
from random import choices, uniform

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
