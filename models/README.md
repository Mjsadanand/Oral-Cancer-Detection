# Models Directory

This directory contains trained machine learning models for oral cancer detection.

## Files

- `oral_cancer_model.h5` - Main trained model (created after training)
- `training_history.png` - Training metrics visualization (created after training)

## Training

To generate the model file, run:

```bash
python train_model.py
```

## Model Info

- **Architecture**: EfficientNetB0 with transfer learning (default)
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Normal vs Cancerous)
- **Format**: HDF5 (.h5)

## Alternative Models

You can train different architectures by modifying `train_model.py`:

```python
model = create_model(model_type='efficientnet')  # default
model = create_model(model_type='resnet50')      # ResNet50
model = create_model(model_type='mobilenet')     # MobileNetV2
model = create_model(model_type='custom')        # Custom CNN
```

## Model Size

Expected model sizes:
- EfficientNetB0: ~20-30 MB
- ResNet50: ~90-100 MB
- MobileNetV2: ~15-20 MB
- Custom CNN: ~5-10 MB
