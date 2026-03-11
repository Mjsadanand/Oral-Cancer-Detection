# Histopathology Image Processing Guide

## 🔬 Overview

This guide covers the updated workflow for oral cancer detection using histopathology images. The system has been optimized specifically for tissue sample analysis with improved preprocessing, data augmentation, and model architecture.

## 📋 Table of Contents

1. [Image Renaming](#1-image-renaming)
2. [Model Training](#2-model-training)
3. [Making Predictions](#3-making-predictions)
4. [Web Application](#4-web-application)
5. [Best Practices](#5-best-practices)

---

## 1. Image Renaming

### Why Rename Images?

Histopathology images often have inconsistent naming patterns like `aug_65_1927`, `OSCC_400x_54`, etc. Standardizing names helps with:
- Better organization
- Easier tracking
- Consistent versioning
- Simplified debugging

### Using the Rename Script

#### Dry Run (Preview Changes)

```bash
python rename_images.py
```

This shows what will be renamed without making any changes.

#### Execute Renaming

```bash
python rename_images.py --execute
```

This will:
- Rename all images in `dataset/train/cancerous/` to `cancer_train_0001.jpg`, `cancer_train_0002.jpg`, etc.
- Rename all images in `dataset/validation/cancerous/` to `cancer_val_0001.jpg`, `cancer_val_0002.jpg`, etc.
- Create backups in `backup_originals/` folder

#### Advanced Options

```bash
# Rename specific directory
python rename_images.py --dir dataset/train/cancerous --prefix oscc --execute

# Execute without backup
python rename_images.py --execute --no-backup

# Custom starting index
python rename_images.py --dir path/to/images --prefix cancer --start-index 100 --execute

# Custom dataset path
python rename_images.py --dataset-path /path/to/dataset --execute
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)

---

## 2. Model Training

### What's Changed for Histopathology?

The training script has been optimized for histopathology images with:

✅ **Enhanced Data Augmentation**
- 180° rotation (histopathology is rotation-invariant)
- Vertical and horizontal flipping
- Brightness variation (mimics staining differences)
- Channel shifting (handles color variations)

✅ **Improved Model Architecture**
- DenseNet121 (recommended for medical imaging)
- EfficientNetB0
- ResNet50
- Custom deeper CNN with batch normalization

✅ **Better Training Strategy**
- Class weight balancing
- AUC metric (better for medical diagnosis)
- TensorBoard logging
- Increased epochs (100) with early stopping

### Training Your Model

#### Step 1: Organize Dataset

Ensure your dataset follows this structure:

```
dataset/
├── train/
│   ├── normal/
│   │   ├── normal_0001.jpg
│   │   ├── normal_0002.jpg
│   │   └── ...
│   └── cancerous/
│       ├── cancer_train_0001.jpg
│       ├── cancer_train_0002.jpg
│       └── ...
└── validation/
    ├── normal/
    │   ├── normal_val_0001.jpg
    │   └── ...
    └── cancerous/
        ├── cancer_val_0001.jpg
        └── ...
```

#### Step 2: Run Training

```bash
# Train with DenseNet (recommended)
python train_model.py
```

The script will:
- Automatically calculate class weights
- Train for up to 100 epochs with early stopping
- Save the best model based on AUC
- Generate training plots
- Log metrics to TensorBoard

#### Step 3: Monitor Training

While training, open another terminal and run:

```bash
tensorboard --logdir=logs/fit
```

Then open http://localhost:6006 in your browser to see:
- Real-time training metrics
- Loss curves
- Accuracy trends
- Precision/Recall graphs

### Model Selection

You can change the model architecture by editing [train_model.py](train_model.py#L156):

```python
model = create_model(model_type='densenet')  # Options: 'densenet', 'efficientnet', 'resnet50', 'custom'
```

**Recommendations:**
- **DenseNet121**: Best for histopathology (recommended)
- **EfficientNetB0**: Good balance of speed and accuracy
- **ResNet50**: Robust, slightly slower
- **Custom CNN**: Fastest, good for limited data

### Configuration Options

Edit the configuration in [train_model.py](train_model.py#L15-L22):

```python
IMG_SIZE = (224, 224)  # Can increase to (299, 299) or (512, 512)
BATCH_SIZE = 16        # Reduce if out of memory
EPOCHS = 100           # Maximum epochs
LEARNING_RATE = 0.0001 # Lower for fine-tuning
```

---

## 3. Making Predictions

### Single Image Prediction

```bash
python predict.py --image path/to/image.jpg
```

**Output:**
```
📊 Prediction Result
============================================================
Image: path/to/image.jpg
Prediction: Cancerous
Confidence: 87.34%
Risk Level: Very High
Raw Score: 0.8734
============================================================
```

### Batch Prediction

```bash
python predict.py --dir dataset/test_images/
```

**Output includes:**
- Individual predictions for each image
- Summary statistics
- Risk distribution
- Average confidence

### Save Results to JSON

```bash
python predict.py --dir dataset/test_images/ --output results.json
```

This creates a JSON file with all predictions:

```json
[
  {
    "image": "dataset/test_images/sample_001.jpg",
    "prediction": "Cancerous",
    "confidence": 87.34,
    "risk_level": "Very High",
    "raw_score": 0.8734
  },
  ...
]
```

### Custom Threshold

```bash
python predict.py --image test.jpg --threshold 0.6
```

Adjust the classification threshold (default: 0.5) to control sensitivity:
- Lower threshold (0.3-0.4): More sensitive, catches more cancers but more false positives
- Higher threshold (0.6-0.7): More specific, fewer false positives but may miss some cases

---

## 4. Web Application

### Starting the Server

```bash
python app.py
```

Then open: http://localhost:8000

### API Endpoints

#### Upload Image for Prediction

**POST** `/api/predict`

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "prediction": "Cancerous",
  "confidence": 87.34,
  "raw_score": 0.8734,
  "risk_level": "Very High",
  "recommendations": [
    "⚠️ URGENT: High confidence of malignancy detected",
    "🏭 Immediate referral to oncology specialist required",
    "📋 Professional histopathological confirmation needed",
    "🏥 Discuss treatment options with oncologist immediately",
    "📞 Schedule biopsy and imaging studies"
  ]
}
```

#### Health Check

**GET** `/api/health`

```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 5. Best Practices

### For Dataset Preparation

1. **Image Quality**
   - Use high-resolution images (minimum 224×224)
   - Ensure proper focus and lighting
   - Consistent staining protocols

2. **Data Organization**
   - Rename images before training
   - Keep backups of original files
   - Maintain consistent naming conventions

3. **Dataset Balance**
   - Aim for similar numbers of normal and cancerous images
   - If imbalanced, the training script will automatically apply class weights
   - Consider data augmentation for minority class

### For Training

1. **Start with Transfer Learning**
   - Use DenseNet or EfficientNet with ImageNet weights
   - Train for at least 50 epochs
   - Monitor validation metrics

2. **Avoid Overfitting**
   - Use data augmentation
   - Apply dropout layers
   - Monitor train vs. validation accuracy gap

3. **Fine-tuning** (Advanced)
   - After initial training, unfreeze base model layers
   - Lower learning rate to 1e-5
   - Train for additional epochs

### For Deployment

1. **Model Validation**
   - Test on separate test set
   - Calculate confusion matrix
   - Measure precision, recall, F1 score

2. **Production Considerations**
   - Monitor prediction latency
   - Log all predictions for audit
   - Implement human review for high-stakes decisions

3. **Legal & Ethical**
   - ⚠️ **This system is for research purposes only**
   - ⚠️ **Not approved for clinical diagnosis**
   - ⚠️ **Always require professional pathologist review**
   - Ensure HIPAA compliance if using patient data

---

## 🔧 Troubleshooting

### Out of Memory Error

Reduce batch size in [train_model.py](train_model.py):
```python
BATCH_SIZE = 8  # or even 4
```

### Model Not Loading

Ensure the model file exists:
```bash
ls -lh models/oral_cancer_model.h5
```

If not, train the model first:
```bash
python train_model.py
```

### Image Format Issues

Convert images to supported formats:
```bash
# Convert all BMP to JPG
for img in dataset/train/cancerous/*.bmp; do
    convert "$img" "${img%.bmp}.jpg"
done
```

### Poor Performance

1. Check dataset quality
2. Increase training epochs
3. Try different model architectures
4. Ensure proper data augmentation
5. Verify image preprocessing matches training

---

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Medical Image Analysis Best Practices](https://www.nature.com/articles/s41746-019-0141-3)
- [Histopathology Image Analysis Guide](https://arxiv.org/abs/1912.08893)

---

## 📝 Summary

### Workflow

1. **Rename images** (optional but recommended)
   ```bash
   python rename_images.py --execute
   ```

2. **Train model**
   ```bash
   python train_model.py
   ```

3. **Test predictions**
   ```bash
   python predict.py --dir dataset/validation/cancerous/
   ```

4. **Deploy web app**
   ```bash
   python app.py
   ```

### Key Files

- `rename_images.py` - Rename and organize images
- `train_model.py` - Train the deep learning model
- `predict.py` - Make batch predictions
- `app.py` - FastAPI web application
- `models/oral_cancer_model.h5` - Trained model weights

---

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. It is **NOT** intended for clinical diagnosis or medical decision-making. Always consult qualified pathologists and oncologists for professional medical advice.

---

**Questions?** Check the [README.md](README.md) or [QUICKSTART.md](QUICKSTART.md) for more information.
