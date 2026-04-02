"""
Batch Prediction Script for Histopathology Images
Use this to make predictions on multiple histopathology images
Optimized for oral cancer detection from tissue samples
"""

import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import json

# Model path
MODEL_PATH = "models/oral_cancer_model.h5"

# Class labels
CLASS_LABELS = {
    0: "Normal",
    1: "Cancerous"
}

def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess histopathology image for prediction"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary (important for histopathology)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize with high-quality resampling
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0,1] - same as training
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"❌ Error preprocessing {image_path}: {e}")
        return None

def predict_single(model, image_path, threshold=0.5):
    """Make prediction on a single histopathology image"""
    
    # Preprocess
    img = preprocess_image(image_path)
    if img is None:
        return None
    
    # Predict
    predictions = model.predict(img, verbose=0)
    raw_score = float(predictions[0][0])
    
    # Determine class
    predicted_class = 1 if raw_score > threshold else 0
    confidence = raw_score if predicted_class == 1 else 1 - raw_score
    
    # Calculate risk level
    if predicted_class == 1:
        if confidence >= 0.85:
            risk_level = "Very High"
        elif confidence >= 0.70:
            risk_level = "High"
        else:
            risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    return {
        "image": str(image_path),
        "prediction": CLASS_LABELS[predicted_class],
        "confidence": round(confidence * 100, 2),
        "risk_level": risk_level,
        "raw_score": raw_score
    }

def predict_batch(model, image_dir, output_file=None):
    """Make predictions on all images in a directory"""
    
    image_dir = Path(image_dir)
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(ext))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"\n📁 Found {len(image_files)} images")
    print("🔄 Processing...\n")
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}", end=" ... ")
        
        result = predict_single(model, image_path)
        
        if result:
            results.append(result)
            print(f"✅ {result['prediction']} ({result['confidence']}%)")
        else:
            print("❌ Failed")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Prediction Summary")
    print("=" * 60)
    
    total = len(results)
    cancerous = len([r for r in results if r['prediction'] == 'Cancerous'])
    normal = total - cancerous
    
    # Calculate average confidence
    avg_confidence = sum(r['confidence'] for r in results) / total if total > 0 else 0
    
    # Count by risk level
    risk_counts = {}
    for r in results:
        risk = r.get('risk_level', 'Unknown')
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    print(f"Total Images: {total}")
    print(f"Normal: {normal} ({normal/total*100:.1f}%)")
    print(f"Cancerous: {cancerous} ({cancerous/total*100:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.1f}%")
    print("\nRisk Distribution:")
    for risk, count in sorted(risk_counts.items()):
        print(f"  {risk}: {count} ({count/total*100:.1f}%)")
    print("=" * 60)
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Oral Cancer Detection - Batch Prediction for Histopathology Images'
    )
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--dir', type=str, help='Path to directory with images')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model()
    if model is None:
        print("\n❌ Please train the model first: python train_model.py")
        return
    
    # Make predictions
    if args.image:
        # Single image prediction
        print(f"\n🔍 Analyzing: {args.image}")
        result = predict_single(model, args.image)
        
        if result:
            print("\n" + "=" * 60)
            print("📊 Prediction Result")
            print("=" * 60)
            print(f"Image: {result['image']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Raw Score: {result['raw_score']:.4f}")
            print("=" * 60)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Result saved to: {args.output}")
    
    elif args.dir:
        # Batch prediction
        predict_batch(model, args.dir, args.output)
    
    else:
        print("❌ Please provide either --image or --dir argument")
        print("\nExamples:")
        print("  Single image: python predict.py --image path/to/image.jpg")
        print("  Batch: python predict.py --dir path/to/images/")
        print("  With output: python predict.py --dir path/to/images/ --output results.json")
        print("  Custom threshold: python predict.py --image test.jpg --threshold 0.6")

if __name__ == "__main__":
    main() 
    