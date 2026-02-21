"""
Oral Cancer Detection - Model Training Script
This script trains a CNN model for binary classification (Normal vs. Cancerous)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Dataset paths - Update these with your dataset location
DATASET_PATH = "dataset"  # Should contain 'train' and 'validation' folders
                          # Each folder should have 'normal' and 'cancerous' subfolders

def create_data_generators():
    """Create data generators with augmentation"""
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling)
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
    Create model architecture
    
    Args:
        model_type: 'efficientnet', 'resnet50', 'mobilenet', or 'custom'
    """
    
    if model_type == 'efficientnet':
        # Transfer learning with EfficientNetB0
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3)
        )
        base_model.trainable = False  # Freeze base model
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
    elif model_type == 'resnet50':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3)
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
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
        
    else:  # Custom CNN
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
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
    model = create_model(model_type='efficientnet')  # Change model type here
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/oral_cancer_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n🚀 Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_generator)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Precision: {val_precision * 100:.2f}%")
    print(f"Validation Recall: {val_recall * 100:.2f}%")
    print(f"F1 Score: {2 * (val_precision * val_recall) / (val_precision + val_recall):.2f}")
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
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
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
