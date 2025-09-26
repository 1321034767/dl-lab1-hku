#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Collection Script for DASC7606 Assignment I
CIFAR-10 Dataset Processing
"""
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import os
import argparse

def setup_environment():
    """Setup random seeds and plotting style"""
    np.random.seed(42)
    tf.random.set_seed(42)
    plt.style.use('ggplot')
    sns.set_palette("husl")
    print("Environment setup complete.")

def load_cifar10():
    """Load CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def get_class_names():
    """Return CIFAR-10 class names"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

def analyze_dataset(data, labels, dataset_name, class_names):
    """Perform comprehensive dataset analysis"""
    print(f"\n{dataset_name} Analysis:")
    print("=" * 50)
    
    # Basic information
    print(f"Number of samples: {len(data)}")
    print(f"Image shape: {data.shape[1:]}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Class distribution
    class_dist = Counter(labels.flatten())
    print("\nClass Distribution:")
    for class_id, count in class_dist.items():
        print(f"  {class_names[class_id]}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Pixel statistics
    print(f"\nPixel Statistics:")
    print(f"  Mean: {data.mean():.3f}")
    print(f"  Std: {data.std():.3f}")
    print(f"  Min: {data.min()}")
    print(f"  Max: {data.max()}")
    
    return class_dist

def visualize_samples(images, labels, class_names, num_samples=10, save_path=None):
    """Visualize sample images"""
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f'{class_names[labels[i][0]]} (Class: {labels[i][0]})')
        plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images saved to {save_path}")
    plt.show()

def plot_class_distribution(labels, class_names, title, save_path=None):
    """Plot class distribution charts"""
    plt.figure(figsize=(12, 6))
    class_counts = Counter(labels.flatten())
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(class_names)), [class_counts[i] for i in range(len(class_names))])
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.title(f'{title} - Class Distribution')
    plt.xticks(range(len(class_names)), range(len(class_names)))
    
    plt.subplot(1, 2, 2)
    plt.pie([class_counts[i] for i in range(len(class_names))], 
            labels=class_names, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'{title} - Class Percentage')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    plt.show()

def analyze_pixel_values(images, title, save_path=None):
    """Analyze RGB channel distributions"""
    plt.figure(figsize=(15, 5))
    flattened_pixels = images.reshape(-1, 3)
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']
    
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(flattened_pixels[:, i], bins=50, color=colors[i], alpha=0.7)
        plt.title(f'{channel_names[i]} Channel Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    
    plt.suptitle(f'{title} - RGB Channel Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pixel analysis plot saved to {save_path}")
    plt.show()
    
    # Print statistics
    print(f"\n{title} Channel Statistics:")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        print(f"  {channel}: Mean={flattened_pixels[:, i].mean():.2f}, "
              f"Std={flattened_pixels[:, i].std():.2f}")

def split_data(x_train, y_train, test_size=0.2, random_state=42):
    """Split data into training and validation sets"""
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
    )
    
    print(f"Final Training set: {x_train_final.shape}")
    print(f"Validation set: {x_val.shape}")
    
    return x_train_final, x_val, y_train_final, y_val

def save_processed_data(x_train, y_train, x_val, y_val, x_test, y_test, class_names, base_path='../data/processed'):
    """Save processed data to files"""
    os.makedirs(base_path, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(base_path, 'x_train.npy'), x_train)
    np.save(os.path.join(base_path, 'y_train.npy'), y_train)
    np.save(os.path.join(base_path, 'x_val.npy'), x_val)
    np.save(os.path.join(base_path, 'y_val.npy'), y_val)
    np.save(os.path.join(base_path, 'x_test.npy'), x_test)
    np.save(os.path.join(base_path, 'y_test.npy'), y_test)
    
    # Save class names
    with open(os.path.join(base_path, 'class_names.txt'), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"Processed data saved to {base_path}")

def generate_summary_report(x_train, y_train, x_val, y_val, x_test, y_test, class_names):
    """Generate final summary report"""
    print("=" * 60)
    print("DATA COLLECTION SUMMARY REPORT")
    print("=" * 60)
    print(f"Dataset: CIFAR-10")
    print(f"Total classes: {len(class_names)}")
    print(f"Original training samples: {len(x_train) + len(x_val)}")
    print(f"Final training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Data type: {x_train.dtype}")
    print(f"Pixel value range: [0, 255]")
    print("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CIFAR-10 Data Collection')
    parser.add_argument('--save-plots', action='store_true', help='Save visualization plots')
    parser.add_argument('--output-dir', default='../data/processed', help='Output directory for processed data')
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    class_names = get_class_names()
    
    # Analyze original datasets
    analyze_dataset(x_train, y_train, "Original Training Set", class_names)
    analyze_dataset(x_test, y_test, "Test Set", class_names)
    
    # Visualizations
    visualize_samples(x_train, y_train, class_names, 
                     save_path=os.path.join(args.output_dir, 'sample_images.png') if args.save_plots else None)
    
    plot_class_distribution(y_train, class_names, "Training Set",
                           save_path=os.path.join(args.output_dir, 'train_class_distribution.png') if args.save_plots else None)
    
    analyze_pixel_values(x_train, "Training Set",
                        save_path=os.path.join(args.output_dir, 'pixel_analysis.png') if args.save_plots else None)
    
    # Split data
    x_train_final, x_val, y_train_final, y_val = split_data(x_train, y_train)
    
    # Analyze split datasets
    analyze_dataset(x_train_final, y_train_final, "Final Training Set", class_names)
    analyze_dataset(x_val, y_val, "Validation Set", class_names)
    
    # Save processed data
    save_processed_data(x_train_final, y_train_final, x_val, y_val, x_test, y_test, class_names, args.output_dir)
    
    # Generate summary
    generate_summary_report(x_train_final, y_train_final, x_val, y_val, x_test, y_test, class_names)

if __name__ == "__main__":
    main()