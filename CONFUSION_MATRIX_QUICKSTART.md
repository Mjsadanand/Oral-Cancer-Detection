# Confusion Matrix - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Confusion Matrix
Choose one of these methods:

#### Option A: Generate During Model Training
```bash
python train_model.py
```
✅ Automatically generates confusion matrix after training completes
✅ Displays metrics in terminal
✅ Saves visualization to `models/confusion_matrix.png`

#### Option B: Generate From Existing Model
```bash
python generate_confusion_matrix.py
```
✅ Works with already-trained models
✅ Generates comprehensive report with visualizations
✅ Shows metrics in terminal and saves image

### Step 3: View in Web UI
1. Run the web application: `python app.py`
2. Open browser: `http://localhost:5000`
3. Click on **"Model Analytics"** tab
4. See confusion matrix + performance metrics

---

## 📊 What You'll See

### In Terminal
```
✅ Confusion matrix saved to 'models/confusion_matrix.png'

📋 CLASSIFICATION REPORT
                precision    recall  f1-score   support
      Normal       0.8889    0.9091    0.8989        44
   Cancerous       0.8800    0.8571    0.8685        46

📊 CONFUSION MATRIX DETAILS
Sensitivity (Recall)          88.89%
Specificity                   90.91%
Accuracy                      88.00%
F1 Score                      0.8989
ROC AUC                       0.9451
```

### In Web UI
- 📈 **Accuracy** meter (green progress bar)
- 🎯 **Sensitivity** - Ability to detect cancer
- ✓ **Specificity** - Ability to identify normal cases
- 📊 **F1 Score** - Overall performance
- 📉 **ROC AUC** - Discrimination ability
- 🖼️ **Confusion Matrix Image** - Visual representation
- 📋 **Matrix Summary** - TP, TN, FP, FN counts

---

## 📖 Understanding the Metrics

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Sensitivity** | TP / (TP + FN) | How often we correctly identify cancer |
| **Specificity** | TN / (TN + FP) | How often we correctly identify normal |
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **F1 Score** | 2×(P×R)/(P+R) | Balance between precision & recall |
| **ROC AUC** | Area under curve | Discriminative ability (0.5=bad, 1.0=perfect) |

---

## 🔧 Customization

### Change Dataset Path
```bash
DATASET_PATH=path/to/dataset python generate_confusion_matrix.py
```

### Save Metrics as JSON
Edit `generate_confusion_matrix.py` to save metrics:
```python
import json
metrics = generate_confusion_matrix_report(model, val_generator, class_labels)
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f)
```

---

## ⚡ API Endpoints

### Get Metrics (JSON)
```bash
curl http://localhost:5000/api/metrics
```

### Get Confusion Matrix (Image)
```bash
curl http://localhost:5000/api/confusion-matrix -o cm.png
```

### Get Metrics with Image (Base64)
```bash
curl http://localhost:5000/api/confusion-matrix/base64
```

---

## ✅ Verification Checklist

- [ ] Dependencies installed: `pip install scikit-learn seaborn`
- [ ] Model trained or existing model available
- [ ] Confusion matrix generated: `python generate_confusion_matrix.py`
- [ ] Image file exists: `ls models/confusion_matrix.png`
- [ ] Web UI shows "Model Analytics" tab
- [ ] Metrics display with values (not dashes)

---

## 🐛 Troubleshooting

### Q: "confusion_matrix.png not found"
```bash
# A: Generate it first
python generate_confusion_matrix.py
```

### Q: "sklearn not found"
```bash
# A: Install scikit-learn
pip install scikit-learn
```

### Q: Metrics show "--" in UI
```bash
# A: Refresh page or run
python generate_confusion_matrix.py
```

### Q: No validation data
```bash
# A: Create validation folder
dataset/validation/normal/
dataset/validation/cancerous/
```

---

## 📚 Full Documentation
See [CONFUSION_MATRIX_GUIDE.md](CONFUSION_MATRIX_GUIDE.md) for detailed information.

---

## 🎯 Key Files
- `generate_confusion_matrix.py` - Standalone generator
- `train_model.py` - Integrated into training
- `api/index.py` - API endpoints
- `static/index.html` - Web UI with analytics tab
- `models/confusion_matrix.png` - Generated visualization

---

**Questions?** Check `CONFUSION_MATRIX_GUIDE.md` for comprehensive documentation!
