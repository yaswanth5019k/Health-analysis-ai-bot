#!/usr/bin/env python3
"""
Create a smaller dataset for faster training
Samples a subset of images from each class
"""

import os
import shutil
import random
from pathlib import Path

def create_small_dataset(samples_per_class=500):
    """Create a smaller dataset with specified samples per class"""
    print(f"📁 Creating small dataset with {samples_per_class} samples per class...")
    
    # Create small dataset directory
    small_dataset_dir = 'small_dataset'
    os.makedirs(small_dataset_dir, exist_ok=True)
    
    # Classes to process
    classes = ['Normal', 'Pneumonia', 'COVID-19']
    
    for class_name in classes:
        source_dir = f'dataset/{class_name}'
        target_dir = f'{small_dataset_dir}/{class_name}'
        
        if os.path.exists(source_dir):
            os.makedirs(target_dir, exist_ok=True)
            
            # Get all image files
            image_files = [f for f in os.listdir(source_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Sample images
            if len(image_files) > samples_per_class:
                selected_files = random.sample(image_files, samples_per_class)
            else:
                selected_files = image_files
            
            # Copy selected files
            for filename in selected_files:
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, filename)
                shutil.copy2(source_path, target_path)
            
            print(f"✅ {class_name}: {len(selected_files)} images (from {len(image_files)} total)")
    
    # Print summary
    print(f"\n📊 Small Dataset Summary:")
    total_images = 0
    for class_name in classes:
        class_dir = f'{small_dataset_dir}/{class_name}'
        if os.path.exists(class_dir):
            image_count = len([f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   {class_name}: {image_count} images")
            total_images += image_count
    
    print(f"   Total: {total_images} images")
    print(f"✅ Small dataset created in '{small_dataset_dir}' directory!")

def create_tiny_dataset(samples_per_class=100):
    """Create a very small dataset for quick testing"""
    print(f"📁 Creating tiny dataset with {samples_per_class} samples per class...")
    
    # Create tiny dataset directory
    tiny_dataset_dir = 'tiny_dataset'
    os.makedirs(tiny_dataset_dir, exist_ok=True)
    
    # Classes to process
    classes = ['Normal', 'Pneumonia', 'COVID-19']
    
    for class_name in classes:
        source_dir = f'dataset/{class_name}'
        target_dir = f'{tiny_dataset_dir}/{class_name}'
        
        if os.path.exists(source_dir):
            os.makedirs(target_dir, exist_ok=True)
            
            # Get all image files
            image_files = [f for f in os.listdir(source_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Sample images
            if len(image_files) > samples_per_class:
                selected_files = random.sample(image_files, samples_per_class)
            else:
                selected_files = image_files
            
            # Copy selected files
            for filename in selected_files:
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, filename)
                shutil.copy2(source_path, target_path)
            
            print(f"✅ {class_name}: {len(selected_files)} images (from {len(image_files)} total)")
    
    # Print summary
    print(f"\n📊 Tiny Dataset Summary:")
    total_images = 0
    for class_name in classes:
        class_dir = f'{tiny_dataset_dir}/{class_name}'
        if os.path.exists(class_dir):
            image_count = len([f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   {class_name}: {image_count} images")
            total_images += image_count
    
    print(f"   Total: {total_images} images")
    print(f"✅ Tiny dataset created in '{tiny_dataset_dir}' directory!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create smaller datasets for faster training')
    parser.add_argument('--size', type=str, default='small', 
                       choices=['small', 'tiny'],
                       help='Dataset size: small (500 per class) or tiny (100 per class)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Custom number of samples per class')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    if args.size == 'small':
        samples = args.samples if args.samples else 500
        create_small_dataset(samples)
    elif args.size == 'tiny':
        samples = args.samples if args.samples else 100
        create_tiny_dataset(samples) 