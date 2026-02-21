# 🚀 Quick Start Guide

## Step 1: Verify Installation

Run the test script to check if everything is set up correctly:

```bash
python test_setup.py
```

This will verify:
- ✅ All required packages are installed
- ✅ Directory structure is correct
- ✅ Required files exist
- ✅ GPU availability (if applicable)

---

## Step 2: Prepare Your Dataset

Create the following folder structure:

```
dataset/
├── train/
│   ├── normal/       # Put normal oral images here
│   └── cancerous/    # Put cancerous oral images here
└── validation/
    ├── normal/       # Put validation normal images here
    └── cancerous/    # Put validation cancerous images here
```

**Tips:**
- Aim for at least 500 images per class
- Use 80% for training, 20% for validation
- Ensure images are clear and properly labeled
- Balance your dataset (similar numbers in each class)

---

## Step 3: Train the Model

Run the training script:

```bash
python train_model.py
```

**What happens:**
- Loads and preprocesses your images
- Trains a deep learning model
- Saves the best model to `models/oral_cancer_model.h5`
- Generates training plots
- Shows accuracy, precision, and recall metrics

**Training Time:**
- With GPU: 10-30 minutes
- Without GPU: 1-3 hours

---

## Step 4: Run the Application

Start the web server:

```bash
python app.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 5: Open in Browser

Open your browser and go to:

```
http://localhost:8000
```

You should see the modern, purple-gradient interface with:
- Upload area for images
- Analysis button
- Results section

---

## Step 6: Test It Out

1. **Upload an Image**
   - Click the upload area or drag & drop
   - Select an oral cavity image

2. **Analyze**
   - Click "Analyze Image"
   - Wait 1-2 seconds

3. **View Results**
   - See prediction (Normal/Cancerous)
   - Check confidence percentage
   - Read recommendations

---

## 🐳 Using Docker (Alternative)

If you prefer Docker:

### Build and Run

```bash
docker-compose up --build
```

### Access the App

```
http://localhost:8000
```

### Stop the App

```bash
docker-compose down
```

---

## 🔧 Troubleshooting

### "Model not found" error
- Make sure you've trained the model (Step 3)
- Check that `models/oral_cancer_model.h5` exists

### "No module named..." error
- Install dependencies: `pip install -r requirements.txt`
- Make sure you're in the correct directory

### Port 8000 already in use
- Stop other applications using port 8000
- Or change the port in `app.py`:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=8080)
  ```

### Out of memory during training
- Reduce batch size in `train_model.py`:
  ```python
  BATCH_SIZE = 16  # or even 8
  ```

### Slow predictions
- Model is using CPU. GPU will speed it up
- Or use a lighter model (MobileNetV2)

---

## 📊 Expected Results

After training, you should see:

```
✅ Training Complete!
==================================================
Validation Accuracy: 92.50%
Validation Precision: 94.20%
Validation Recall: 89.80%
F1 Score: 0.92
==================================================
```

**Good Performance:**
- Accuracy: > 90%
- Precision: > 90%
- Recall: > 85%

**If lower:**
- Get more training data
- Train for more epochs
- Try different model architectures

---

## 🎯 Next Steps

Once your app is running:

1. **Test with various images**
   - Test both normal and cancerous images
   - Verify predictions make sense

2. **Improve the model**
   - Collect more data
   - Fine-tune hyperparameters
   - Try ensemble methods

3. **Customize the UI**
   - Edit `static/index.html`
   - Change colors, text, layout
   - Add your branding

4. **Deploy to production**
   - Use Docker for deployment
   - Deploy to cloud (AWS, GCP, Azure)
   - Add authentication if needed

---

## 📚 Learn More

- [Full Documentation](README.md)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

---

## 🆘 Need Help?

If you're stuck:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Run `python test_setup.py` to diagnose issues
3. Review the console output for error messages
4. Open an issue on GitHub

---

**Good luck with your project! 🚀**
