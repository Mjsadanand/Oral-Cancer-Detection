# Confusion Matrix Implementation Guide

## Overview
The confusion matrix feature provides detailed model performance metrics including sensitivity, specificity, accuracy, F1 score, and ROC AUC. It can be displayed both in the shell/terminal and in the web UI.

## Components Added

### 1. **Backend Training Script** (`train_model.py`)
- **New Function**: `generate_confusion_matrix(model, val_generator)`
- **Features**:
  - Generates predictions on validation data
  - Creates confusion matrix visualization
  - Displays confusion matrix as both counts and percentages
  - Calculates diagnostic metrics (sensitivity, specificity, PPV, NPV)
  - Prints detailed classification report to console
  - **Auto-called** after model training completes
  - **Output**: Saves visualization to `models/confusion_matrix.png`

**Usage**: The confusion matrix is automatically generated when you run:
```bash
python train_model.py
```

### 2. **Standalone Confusion Matrix Generator** (`generate_confusion_matrix.py`)
- **Purpose**: Generate confusion matrix on demand without retraining
- **Features**:
  - Loads pre-trained model
  - Evaluates on validation dataset
  - Generates comprehensive metrics report
  - Creates visualization with:
    - Confusion matrix (counts)
    - Normalized confusion matrix (percentages)
    - ROC curve with AUC
    - Detailed metrics summary
  - Saves as `models/confusion_matrix.png`
  - Prints all metrics to terminal

**Usage**:
```bash
python generate_confusion_matrix.py
```

**Optional arguments**:
```bash
python generate_confusion_matrix.py --model models/oral_cancer_model.h5 --dataset dataset/
```

### 3. **API Endpoints** (`api/index.py`)

#### `/api/metrics` (GET)
Returns detailed model metrics as JSON:
```json
{
  "status": "success",
  "performance": {
    "sensitivity": 0.92,      // TP / (TP + FN) - Ability to detect cancer
    "specificity": 0.88,      // TN / (TN + FP) - Ability to identify normal
    "accuracy": 0.90,         // (TP + TN) / Total
    "precision": 0.88,        // TP / (TP + FP)
    "recall": 0.92,           // Same as sensitivity
    "f1_score": 0.89,
    "roc_auc": 0.95
  },
  "confusion_matrix": {
    "true_positives": 46,
    "true_negatives": 44,
    "false_positives": 6,
    "false_negatives": 4,
    "total_samples": 100
  }
}
```

#### `/api/confusion-matrix` (GET)
Returns confusion matrix image as PNG file.

#### `/api/confusion-matrix/base64` (GET)
Returns base64-encoded confusion matrix image (useful for web display):
```json
{
  "status": "success",
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "format": "base64"
}
```

### 4. **Web UI** (`static/index.html`)

#### New Tab: "Model Analytics"
Located next to "Image Prediction" tab with tabs for:

1. **Confusion Matrix Visualization**
   - Shows the actual confusion matrix image
   - Interactive refresh button
   - Responsive layout

2. **Performance Metrics Cards**
   - Real-time metric values
   - Color-coded progress bars
   - Interpretable descriptions
   - Metrics included:
     - Accuracy (green)
     - Sensitivity/Recall (blue)
     - Specificity (indigo)
     - F1 Score (orange)
     - ROC AUC (pink)

3. **Matrix Summary**
   - True Positives (TP)
   - True Negatives (TN)
   - False Positives (FP)
   - False Negatives (FN)

## Metrics Explanation

### Key Metrics
- **Sensitivity (Recall)**: TP / (TP + FN)
  - Ability to correctly identify cancer cases
  - High sensitivity means few false negatives
  - Critical for medical diagnosis

- **Specificity**: TN / (TN + FP)
  - Ability to correctly identify normal cases
  - High specificity means few false positives
  - Important to avoid unnecessary treatments

- **Accuracy**: (TP + TN) / Total
  - Overall correctness of predictions
  - Can be misleading with imbalanced data

- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - Harmonic mean of precision and recall
  - Good for imbalanced datasets

- **ROC AUC**: Area Under the ROC Curve
  - Measures discriminative ability across all thresholds
  - 0.5 = random classifier, 1.0 = perfect classifier

## Shell Output Example

When you run `python generate_confusion_matrix.py`:

```
======================================================================
🏥 Oral Cancer Detection - Confusion Matrix Generator
======================================================================

✅ Model loaded from models/oral_cancer_model.h5
✅ Class labels loaded: {0: 'Normal', 1: 'Cancerous'}
✅ Validation data loaded: 100 samples

======================================================================
🔄 Generating Predictions on Validation Dataset...
======================================================================

✅ Predictions completed
   Total samples: 100
   Correct predictions: 90
   Incorrect predictions: 10

✅ Confusion matrix visualization saved to 'models/confusion_matrix.png'

======================================================================
📋 CLASSIFICATION REPORT
======================================================================
              precision    recall  f1-score   support

      Normal       0.8889    0.9091    0.8989        44
   Cancerous       0.8800    0.8571    0.8685        46

    accuracy                 0.8800       100
   macro avg       0.8844    0.8831    0.8837       100
weighted avg       0.8843    0.8800    0.8821       100

======================================================================
📊 CONFUSION MATRIX DETAILS
======================================================================
Metric                         Value
--------------------------------------------------
True Negatives (TN)              40
False Positives (FP)              4
False Negatives (FN)              6
True Positives (TP)              40

--------------------------------------------------
Sensitivity (Recall)          88.89%
Specificity                   90.91%
PPV (Precision)               91.11%
NPV                           87.04%
Accuracy                      88.00%
F1 Score                      0.8989
ROC AUC                       0.9451
======================================================================
```

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

The updated requirements now include:
- `scikit-learn>=1.6.0` - For confusion matrix and metrics
- `seaborn>=0.13.0` - For improved visualizations

### 2. Train Model (generates confusion matrix automatically)
```bash
python train_model.py
```

### 3. Generate Confusion Matrix On Demand
```bash
python generate_confusion_matrix.py
```

### 4. View in Web UI
1. Run the application: `python app.py`
2. Navigate to http://localhost:5000
3. Click on the "Model Analytics" tab
4. View confusion matrix and performance metrics

## Directory Structure
```
models/
├── oral_cancer_model.h5          # Trained model
├── confusion_matrix.png          # Generated confusion matrix visualization
├── training_history.png          # Training history plots
└── class_indices.json           # Class mapping
```

## Demo Mode
If confusion matrix hasn't been generated yet:
- API returns demo metrics
- UI shows placeholder with instructions
- Instructions to run: `python generate_confusion_matrix.py`

## Customization

### Change Confusion Matrix Visualization
Edit `generate_confusion_matrix.py` to modify:
- Figure size: `figsize=(20, 12)`
- Color schemes: `cmap='Blues'`
- DPI: `dpi=300`
- Font sizes in the visualization

### Customize Metrics Display
Edit `static/index.html` to:
- Add more metric cards
- Change color schemes
- Modify layout for different screen sizes

## Troubleshooting

### Confusion Matrix Not Displaying
1. Ensure model is trained: `python train_model.py`
2. Run: `python generate_confusion_matrix.py`
3. Check file exists: `ls models/confusion_matrix.png`
4. Refresh the web page

### Import Errors
```bash
# If sklearn not found:
pip install scikit-learn

# If seaborn not found:
pip install seaborn
```

### Missing Validation Data
The script automatically uses training data if validation directory is not found. Create validation directory for better evaluation:
```
dataset/
├── train/
│   ├── normal/
│   └── cancerous/
└── validation/
    ├── normal/
    └── cancerous/
```

## Best Practices

1. **Always use validation data** - Separate from training data
2. **Monitor sensitivity first** - For medical diagnosis, catching all positives is critical
3. **Balance sensitivity and specificity** - Adjust decision threshold if needed
4. **Review on multiple batches** - Don't rely on single evaluation
5. **Document metrics** - Save metrics.json for reproducibility

## Integration with CI/CD
Add to your deployment pipeline:
```bash
# Generate metrics after model training
python generate_confusion_matrix.py

# Save metrics to file
python -c "import json; 
import generate_confusion_matrix as cm; 
metrics = cm.generate_confusion_matrix_report(model, val_gen, labels);
with open('models/metrics.json', 'w') as f: json.dump(metrics, f)"
```

## References
- [Confusion Matrix - Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Medical Testing Metrics](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
