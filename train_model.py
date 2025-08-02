#!/usr/bin/env python3
"""
Medical AI Model Training Script
Trains a CNN model for medical image classification using your datasets
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import argparse
import shutil

class MedicalAITrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def create_model(self):
        """Create the CNN model architecture"""
        print("🔧 Creating model architecture...")
        
        if self.config['model_type'] == 'custom_cnn':
            # Custom CNN architecture
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(self.config['classes']), activation='softmax')
            ])
            
        elif self.config['model_type'] == 'resnet50':
            # Transfer learning with ResNet50
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(self.config['classes']), activation='softmax')
            ])
            
        elif self.config['model_type'] == 'efficientnet':
            # Transfer learning with EfficientNet
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(self.config['classes']), activation='softmax')
            ])
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created successfully!")
        print(f"📊 Model summary:")
        self.model.summary()
        
    def prepare_data_generators(self):
        """Prepare data generators for training and validation"""
        print("📁 Preparing data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            self.config['data_dir'],
            target_size=(224, 224),
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        self.val_generator = val_datagen.flow_from_directory(
            self.config['data_dir'],
            target_size=(224, 224),
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"✅ Data generators prepared!")
        print(f"📊 Training samples: {self.train_generator.samples}")
        print(f"📊 Validation samples: {self.val_generator.samples}")
        print(f"📊 Classes: {list(self.train_generator.class_indices.keys())}")
        
        # Update config with actual class names
        self.config['classes'] = list(self.train_generator.class_indices.keys())
        
    def setup_callbacks(self):
        """Setup training callbacks"""
        print("⚙️ Setting up training callbacks...")
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        self.callbacks = [checkpoint, early_stopping, reduce_lr]
        print("✅ Callbacks configured!")
        
    def train(self):
        """Train the model"""
        print("🚀 Starting model training...")
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.config['epochs'],
            validation_data=self.val_generator,
            callbacks=self.callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        
    def save_model(self):
        """Save the trained model and configuration"""
        print("💾 Saving model and configuration...")
        
        # Save the final model
        self.model.save('model.h5')
        
        # Save configuration
        config_to_save = {
            'classes': self.config['classes'],
            'input_shape': (224, 224, 3),
            'model_type': self.config['model_type'],
            'training_history': {
                'final_accuracy': float(self.history.history['accuracy'][-1]),
                'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
                'final_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1])
            }
        }
        
        with open('model_config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print("✅ Model and configuration saved!")
        
    def plot_training_history(self):
        """Plot training history"""
        print("📈 Plotting training history...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training history plotted and saved!")

def prepare_covid_dataset():
    """Prepare COVID-19 dataset for training"""
    print("🦠 Preparing COVID-19 dataset...")
    
    # Create dataset directory
    os.makedirs('dataset', exist_ok=True)
    
    # Map COVID-19 dataset classes to our expected classes
    class_mapping = {
        'Normal': 'Normal',
        'COVID': 'COVID-19',
        'Lung_Opacity': 'Pneumonia',
        'Viral Pneumonia': 'Pneumonia'
    }
    
    # Copy files to dataset directory
    for old_class, new_class in class_mapping.items():
        source_dir = f'COVID-19_Radiography_Dataset/{old_class}'
        target_dir = f'dataset/{new_class}'
        
        if os.path.exists(source_dir):
            os.makedirs(target_dir, exist_ok=True)
            
            # Check if images are in a subdirectory
            images_dir = os.path.join(source_dir, 'images')
            if os.path.exists(images_dir):
                source_dir = images_dir
            
            # Copy all images from source to target
            copied_count = 0
            for filename in os.listdir(source_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_path = os.path.join(source_dir, filename)
                    target_path = os.path.join(target_dir, filename)
                    shutil.copy2(source_path, target_path)
                    copied_count += 1
            
            print(f"✅ Copied {copied_count} images from {old_class} to {new_class}")
    
    print("✅ COVID-19 dataset prepared!")

def prepare_chest_xray_dataset():
    """Prepare chest X-ray dataset for training"""
    print("🫁 Preparing chest X-ray dataset...")
    
    # Create dataset directory
    os.makedirs('dataset', exist_ok=True)
    
    # Map chest X-ray classes to our expected classes
    class_mapping = {
        'NORMAL': 'Normal',
        'PNEUMONIA': 'Pneumonia'
    }
    
    # Copy files from train directory
    for old_class, new_class in class_mapping.items():
        source_dir = f'chest_xray/train/{old_class}'
        target_dir = f'dataset/{new_class}'
        
        if os.path.exists(source_dir):
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy all images from source to target
            for filename in os.listdir(source_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_path = os.path.join(source_dir, filename)
                    target_path = os.path.join(target_dir, filename)
                    shutil.copy2(source_path, target_path)
            
            print(f"✅ Copied {len([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])} images from {old_class} to {new_class}")
    
    print("✅ Chest X-ray dataset prepared!")

def combine_datasets():
    """Combine both datasets for training"""
    print("🔗 Combining datasets...")
    
    # Create combined dataset directory
    os.makedirs('dataset', exist_ok=True)
    
    # Prepare both datasets
    prepare_covid_dataset()
    prepare_chest_xray_dataset()
    
    # Count total images per class
    print("\n📊 Dataset Summary:")
    for class_name in ['Normal', 'Pneumonia', 'COVID-19']:
        class_dir = f'dataset/{class_name}'
        if os.path.exists(class_dir):
            image_count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   {class_name}: {image_count} images")
    
    print("✅ Datasets combined successfully!")

def main():
    parser = argparse.ArgumentParser(description='Train Medical AI Model')
    parser.add_argument('--dataset', type=str, default='combined',
                       choices=['covid', 'chest_xray', 'combined', 'tiny', 'small'],
                       help='Which dataset to use for training')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Custom data directory path')
    parser.add_argument('--model_type', type=str, default='resnet50',
                       choices=['custom_cnn', 'resnet50', 'efficientnet'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--prepare_data', action='store_true',
                       help='Prepare dataset structure')
    
    args = parser.parse_args()
    
    if args.prepare_data:
        if args.dataset == 'covid':
            prepare_covid_dataset()
        elif args.dataset == 'chest_xray':
            prepare_chest_xray_dataset()
        elif args.dataset == 'combined':
            combine_datasets()
        return
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif args.dataset == 'tiny':
        data_dir = 'tiny_dataset'
    elif args.dataset == 'small':
        data_dir = 'small_dataset'
    else:
        # Prepare dataset based on choice
        if args.dataset == 'covid':
            prepare_covid_dataset()
        elif args.dataset == 'chest_xray':
            prepare_chest_xray_dataset()
        elif args.dataset == 'combined':
            combine_datasets()
        data_dir = 'dataset'
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory '{data_dir}' not found!")
        print("💡 Use --prepare_data to prepare the dataset first")
        print("💡 Or use --data_dir to specify a custom path")
        return
    
    # Configuration
    config = {
        'data_dir': data_dir,
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'classes': []  # Will be populated from data
    }
    
    # Initialize trainer
    trainer = MedicalAITrainer(config)
    
    # Prepare data
    trainer.prepare_data_generators()
    
    # Create model
    trainer.create_model()
    
    # Setup callbacks
    trainer.setup_callbacks()
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Plot history
    trainer.plot_training_history()
    
    print("🎉 Training completed successfully!")
    print("📁 Files created:")
    print("   - model.h5 (trained model)")
    print("   - model_config.json (model configuration)")
    print("   - training_history.png (training plots)")
    print("   - best_model.h5 (best model during training)")

if __name__ == "__main__":
    main() 