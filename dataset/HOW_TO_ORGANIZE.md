# Dataset Organization Guide

## ✅ Folders Created!

Your dataset folders have been created at: `V:\A\Soumya\dataset\`

## 📁 How to Organize Your Images:

### 1. **Training Images** (80% of your data)

**Normal Images** → Place in:
```
V:\A\Soumya\dataset\train\normal\
```
- Name them: `normal_001.jpg`, `normal_002.jpg`, etc.
- Or any naming: `img1.jpg`, `photo.png`, etc.

**Cancerous Images** → Place in:
```
V:\A\Soumya\dataset\train\cancerous\
```
- Name them: `cancer_001.jpg`, `cancer_002.jpg`, etc.
- Or any naming you prefer

### 2. **Validation Images** (20% of your data)

**Normal Images** → Place in:
```
V:\A\Soumya\dataset\validation\normal\
```

**Cancerous Images** → Place in:
```
V:\A\Soumya\dataset\validation\cancerous\
```

## 💡 **Important Tips:**

### Image Requirements:
- ✅ **Formats**: JPG, JPEG, PNG
- ✅ **Size**: Any size (will be resized to 224x224)
- ✅ **Minimum**: At least 100 images per class (500+ recommended)
- ✅ **Balance**: Similar number of normal and cancerous images

### Data Split:
- **Training**: 80% of total images
- **Validation**: 20% of total images

Example:
- If you have 1000 images total (500 normal, 500 cancerous):
  - Training: 400 normal + 400 cancerous = 800 images
  - Validation: 100 normal + 100 cancerous = 200 images

## 📊 Example Structure:

```
dataset/
├── train/                    (800 images total)
│   ├── normal/              (400 images)
│   │   ├── normal_001.jpg
│   │   ├── normal_002.jpg
│   │   └── ... (398 more)
│   └── cancerous/           (400 images)
│       ├── cancer_001.jpg
│       ├── cancer_002.jpg
│       └── ... (398 more)
└── validation/              (200 images total)
    ├── normal/              (100 images)
    │   ├── normal_val_001.jpg
    │   └── ... (99 more)
    └── cancerous/           (100 images)
        ├── cancer_val_001.jpg
        └── ... (99 more)
```

## 🚀 **Next Steps:**

1. **Copy your images** into the appropriate folders
2. **Verify your setup**:
   ```bash
   V:/A/Soumya/.venv/Scripts/python.exe test_setup.py
   ```
3. **Train the model**:
   ```bash
   V:/A/Soumya/.venv/Scripts/python.exe train_model.py
   ```

## ⚠️ **Common Mistakes to Avoid:**

❌ Don't mix normal and cancerous images in the same folder
❌ Don't put images directly in `train/` or `validation/` folders
❌ Don't use very small datasets (less than 100 images per class)
❌ Don't forget to split your data into train/validation

✅ Each image should be in **one folder only** based on its classification
✅ Use clear, consistent naming
✅ Check image quality - blurry images reduce accuracy

## 📝 **File Naming Examples:**

**Option 1 - Sequential:**
- `normal_001.jpg`, `normal_002.jpg`, ...
- `cancer_001.jpg`, `cancer_002.jpg`, ...

**Option 2 - Descriptive:**
- `patient_001_normal.jpg`
- `patient_002_cancer.jpg`

**Option 3 - Original Names:**
- `IMG_20240101_001.jpg`
- `scan_123.png`

Any naming works - the folder determines the classification, not the filename!

---

**Ready to train?** Once your images are in place, run:
```bash
V:/A/Soumya/.venv/Scripts/python.exe train_model.py
```
