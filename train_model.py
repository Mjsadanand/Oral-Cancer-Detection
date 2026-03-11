"""
Oral Cancer Detection - Model Training Script (Histopathology Optimized)
This script trains a CNN model for binary classification (Normal vs. Cancerous)
Specifically optimized for histopathology images
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2, DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import datetime

# Configuration for Histopathology Images
IMG_SIZE = (224, 224)  # Can be increased to (299, 299) or (512, 512) for better results
BATCH_SIZE = 16  # Reduced for larger images and complex models
EPOCHS = 100  # Increased for better convergence
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning

# Dataset paths - Update these with your dataset location
DATASET_PATH = "dataset"  # Should contain 'train' and 'validation' folders
                          # Each folder should have 'normal' and 'cancerous' subfolders

def create_data_generators():
    """Create data generators with histopathology-specific augmentation"""
    
    # Training data augmentation - optimized for histopathology
    # Histopathology images benefit from extensive augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180,  # Full rotation for histopathology
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Important for histopathology
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.85, 1.15],  # Staining variation
        channel_shift_range=0.1,  # Color variation in staining
        fill_mode='reflect'  # Better for medical images
    )
    
    # Validation data (only rescaling - no augmentation for validation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',  # For binary classification
        shuffle=True
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator

def create_model(model_type='efficientnet'):
    """
    Create model architecture optimized for histopathology images
    
    Args:
        model_type: 'efficientnet', 'densenet', 'resnet50', 'mobilenet', or 'custom'
    """
    
    if model_type == 'efficientnet':
        # Transfer learning with EfficientNetB0 - good for histopathology
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3)
        )
        base_model.trainable = False  # Initially freeze, can unfreeze later for fine-tuning
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
    
    elif model_type == 'densenet':
        # DenseNet121 - excellent for medical imaging
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3)
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
    elif model_type == 'resnet50':
        # ResNet50 - robust architecture for histopathology
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3)
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
    elif model_type == 'mobilenet':
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3)
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
    else:  # Custom CNN - deeper architecture for histopathology
        model = models.Sequential([
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.2),
            
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.2),
            
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),
            
            # Block 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),
            
            # Classifier
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
    
    return model

def train_model():
    """Main training function"""
    
    print("=" * 60)
    print("🏥 Oral Cancer Detection - Model Training")
    print("=" * 60)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Load data
    print("\n📁 Loading dataset...")
    train_generator, val_generator = create_data_generators()
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"📊 Classes: {train_generator.class_indices}")
    
    # Create model
    print("\n🏗️  Building model...")
    # Choose: 'efficientnet', 'densenet', 'resnet50', 'mobilenet', or 'custom'
    # DenseNet and EfficientNet work well for histopathology
    model = create_model(model_type='densenet')  # Recommended for histopathology
    
    # Compile model with class weights if data is imbalanced
    # Calculate class weights based on your dataset
    total_samples = train_generator.samples
    # For binary classification, adjust if needed
    pos_samples = sum(train_generator.labels)
    neg_samples = total_samples - pos_samples
    
    if pos_samples > 0 and neg_samples > 0:
        weight_for_0 = (1 / neg_samples) * (total_samples / 2.0)
        weight_for_1 = (1 / pos_samples) * (total_samples / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"\n⚖️  Class weights: {class_weight}")
    else:
        class_weight = None
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        ModelCheckpoint(
            'models/oral_cancer_model.h5',
            monitor='val_auc',  # AUC is better metric for medical imaging
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience for histopathology
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Train model
    print("\n🚀 Starting training...")
    print(f"📊 TensorBoard logs: {log_dir}")
    print("   Run: tensorboard --logdir=logs/fit\n")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight,  # Handle class imbalance
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    results = model.evaluate(val_generator)
    
    # Extract metrics
    val_loss = results[0]
    val_accuracy = results[1]
    val_precision = results[2]
    val_recall = results[3]
    val_auc = results[4]
    
    # Calculate F1 Score
    if val_precision + val_recall > 0:
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    else:
        f1_score = 0.0
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Precision: {val_precision * 100:.2f}%")
    print(f"Validation Recall: {val_recall * 100:.2f}%")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("=" * 60)
    
    # Plot training history
    plot_history(history)
    
    # Save model
    model.save('models/oral_cancer_model_final.h5')
    print("\n💾 Model saved to 'models/oral_cancer_model.h5'")
    
    return model, history

def plot_history(history):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # AUC
    if 'auc' in history.history:
        axes[1, 1].plot(history.history['auc'], label='Train AUC')
        axes[1, 1].plot(history.history['val_auc'], label='Val AUC')
        axes[1, 1].set_title('Model AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("📈 Training plots saved to 'models/training_history.png'")
    plt.show()

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset folder not found!")
        print(f"Please create a '{DATASET_PATH}' folder with the following structure:")
        print("""
        dataset/
        ├── train/
        │   ├── normal/
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   └── cancerous/
        │       ├── image1.jpg
        │       ├── image2.jpg
        │       └── ...
        └── validation/
            ├── normal/
            │   ├── image1.jpg
            │   └── ...
            └── cancerous/
                ├── image1.jpg
                └── ...
        """)
    else:
        model, history = train_model()
