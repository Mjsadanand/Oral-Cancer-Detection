"""
Confusion Matrix Generator for Oral Cancer Detection Model
Generates and visualizes confusion matrix from validation dataset
Can be run independently without retraining the model
"""

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# Configuration
MODEL_PATH = "models/oral_cancer_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Dataset paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "dataset"


def resolve_dataset_path():
    """Resolve dataset path"""
    candidates = [
        (Path.cwd() / DEFAULT_DATASET_PATH),
        (BASE_DIR / DEFAULT_DATASET_PATH),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_model():
    """Load trained model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def load_class_labels():
    """Load class labels"""
    class_indices_file = Path(CLASS_INDICES_PATH)
    if not class_indices_file.exists():
        print(f"⚠️  Class mapping file not found. Using default labels.")
        return {0: "Normal", 1: "Cancerous"}
    
    try:
        with open(class_indices_file, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        labels = {int(idx): str(name).title() for name, idx in class_indices.items()}
        print(f"✅ Class labels loaded: {labels}")
        return labels
    except Exception as e:
        print(f"⚠️  Error loading labels: {e}")
        return {0: "Normal", 1: "Cancerous"}


def create_val_generator():
    """Create validation data generator"""
    dataset_root = resolve_dataset_path()
    validation_dir = dataset_root / "validation"
    
    if not validation_dir.exists():
        print(f"⚠️  Validation directory not found at {validation_dir}")
        print("    Using training data for evaluation instead.")
        validation_dir = dataset_root / "train"
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    val_generator = val_datagen.flow_from_directory(
        str(validation_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"✅ Validation data loaded: {val_generator.samples} samples")
    return val_generator


def generate_confusion_matrix_report(model, val_generator, class_labels):
    """
    Generate confusion matrix and detailed report
    """
    print("\n" + "=" * 70)
    print("🔄 Generating Predictions on Validation Dataset...")
    print("=" * 70)
    
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    # Reset generator
    val_generator.reset()
    
    # Get all predictions
    for images, labels in val_generator:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions.flatten())
        y_pred.extend((predictions > 0.5).astype(int).flatten())
        y_true.extend(labels)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get class names
    class_names = [class_labels.get(i, f"Class_{i}") for i in sorted(class_labels.keys())]
    
    print(f"\n✅ Predictions completed")
    print(f"   Total samples: {len(y_true)}")
    print(f"   Correct predictions: {np.sum(y_true == y_pred)}")
    print(f"   Incorrect predictions: {np.sum(y_true != y_pred)}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Confusion Matrix (counts)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax1,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix - Counts', fontsize=13, fontweight='bold')
    
    # Plot 2: Confusion Matrix (normalized percentage)
    ax2 = fig.add_subplot(gs[0, 1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax2,
                cbar_kws={'label': 'Percentage'},
                annot_kws={'size': 12, 'weight': 'bold'})
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix - Normalized (%)', fontsize=13, fontweight='bold')
    
    # Plot 3: ROC Curve
    ax3 = fig.add_subplot(gs[1, :])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax3.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax3.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax3.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax3.legend(loc="lower right", fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Metrics Summary (as text)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    metrics_text = f"""
    DETAILED METRICS SUMMARY
    {'=' * 80}
    
    CONFUSION MATRIX VALUES:
    • True Negatives (TN):   {tn:>6}        • False Positives (FP): {fp:>6}
    • False Negatives (FN):  {fn:>6}        • True Positives (TP):  {tp:>6}
    
    DIAGNOSTIC PERFORMANCE:
    • Sensitivity (Recall):  {sensitivity*100:>6.2f}%  (TP / (TP + FN)) - Ability to detect cancer
    • Specificity:           {specificity*100:>6.2f}%  (TN / (TN + FP)) - Ability to identify normal cases
    • PPV (Precision):       {ppv*100:>6.2f}%  (TP / (TP + FP)) - Probability prediction is correct when positive
    • NPV:                   {npv*100:>6.2f}%  (TN / (TN + FN)) - Probability prediction is correct when negative
    
    OVERALL PERFORMANCE:
    • Accuracy:              {accuracy*100:>6.2f}%  ((TP + TN) / Total)
    • F1 Score:              {f1_score:>6.4f}    (Harmonic mean of Precision & Recall)
    • ROC AUC:               {roc_auc:>6.4f}    (Area Under ROC Curve)
    
    {'=' * 80}
    """
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✅ Confusion matrix visualization saved to 'models/confusion_matrix.png'")
    plt.show()
    
    # Print detailed classification report
    print("\n" + "=" * 70)
    print("📋 CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Print confusion matrix details
    print("=" * 70)
    print("📊 CONFUSION MATRIX DETAILS")
    print("=" * 70)
    print(f"{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'True Negatives (TN)':<30} {tn:<20}")
    print(f"{'False Positives (FP)':<30} {fp:<20}")
    print(f"{'False Negatives (FN)':<30} {fn:<20}")
    print(f"{'True Positives (TP)':<30} {tp:<20}")
    print()
    print(f"{'Sensitivity (Recall)':<30} {sensitivity*100:.2f}%")
    print(f"{'Specificity':<30} {specificity*100:.2f}%")
    print(f"{'PPV (Precision)':<30} {ppv*100:.2f}%")
    print(f"{'NPV':<30} {npv*100:.2f}%")
    print(f"{'Accuracy':<30} {accuracy*100:.2f}%")
    print(f"{'F1 Score':<30} {f1_score:.4f}")
    print(f"{'ROC AUC':<30} {roc_auc:.4f}")
    print("=" * 70)
    
    return {
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'class_names': class_names
    }


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix for trained model')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to trained model')
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET_PATH), help='Path to dataset')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🏥 Oral Cancer Detection - Confusion Matrix Generator")
    print("=" * 70)
    
    # Load model
    model = load_model()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Load class labels
    class_labels = load_class_labels()
    
    # Create validation generator
    val_generator = create_val_generator()
    
    # Generate confusion matrix report
    metrics = generate_confusion_matrix_report(model, val_generator, class_labels)
    
    print("\n✅ Confusion matrix generation completed!")


if __name__ == "__main__":
    main()
