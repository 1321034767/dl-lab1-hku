#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Augmentation Script for DASC7606 Assignment I
CIFAR-10 Dataset Augmentation and Balancing
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import os
import argparse

# 数据增强
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

# 设置
plt.style.use('ggplot')
sns.set_palette("husl")
np.random.seed(42)

class DataAugmentor:
    """数据增强和平衡处理器"""
    
    def __init__(self, data_path: str = '../data/processed'):
        self.data_path = data_path
        self.class_names = []
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None
        
    def load_data(self) -> None:
        """加载预处理数据"""
        print("Loading processed data...")
        
        try:
            self.x_train = np.load(os.path.join(self.data_path, 'x_train.npy'))
            self.y_train = np.load(os.path.join(self.data_path, 'y_train.npy'))
            self.x_val = np.load(os.path.join(self.data_path, 'x_val.npy'))
            self.y_val = np.load(os.path.join(self.data_path, 'y_val.npy'))
            self.x_test = np.load(os.path.join(self.data_path, 'x_test.npy'))
            self.y_test = np.load(os.path.join(self.data_path, 'y_test.npy'))
            
            with open(os.path.join(self.data_path, 'class_names.txt'), 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            print("Data loaded successfully!")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Please run data collection script first: {e}")
    
    def analyze_class_distribution(self, labels: np.ndarray, title: str) -> Dict[int, int]:
        """分析类别分布"""
        labels_flat = labels.flatten()
        class_dist = Counter(labels_flat)
        
        print(f"\n{title} Class Distribution:")
        print("-" * 40)
        for class_id in range(len(self.class_names)):
            count = class_dist.get(class_id, 0)
            percentage = (count / len(labels_flat)) * 100
            print(f"{self.class_names[class_id]:<12}: {count:>5} samples ({percentage:5.1f}%)")
        
        return class_dist
    
    def create_augmentation_generators(self) -> Dict[str, ImageDataGenerator]:
        """创建增强生成器"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for augmentation")
        
        return {
            'base': ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                fill_mode='nearest'
            ),
            'strong': ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            ),
            'flip_only': ImageDataGenerator(horizontal_flip=True),
            'none': ImageDataGenerator()
        }
    
    def visualize_augmentations(self, augmentation_type: str, 
                               num_samples: int = 5, save_path: Optional[str] = None) -> None:
        """可视化增强效果"""
        generators = self.create_augmentation_generators()
        generator = generators[augmentation_type]
        
        plt.figure(figsize=(15, 8))
        plt.suptitle(f'{augmentation_type} Augmentation Examples', fontsize=16)
        
        sample_indices = np.random.choice(len(self.x_train), num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            original_image = self.x_train[idx]
            original_label = self.y_train[idx][0]
            
            augmented_images = []
            for _ in range(4):
                aug_img = generator.random_transform(original_image.astype('float32'))
                augmented_images.append(aug_img)
            
            plt.subplot(num_samples, 5, i*5 + 1)
            plt.imshow(original_image.astype('uint8'))
            plt.title(f'Original\n{self.class_names[original_label]}')
            plt.axis('off')
            
            for j, aug_img in enumerate(augmented_images):
                plt.subplot(num_samples, 5, i*5 + j + 2)
                plt.imshow(aug_img.astype('uint8'))
                plt.title(f'Aug {j+1}')
                plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def balance_classes(self, strategy: str = 'oversample') -> Tuple[np.ndarray, np.ndarray]:
        """平衡类别分布"""
        y_flat = self.y_train.flatten()
        
        if strategy == 'oversample':
            class_counts = Counter(y_flat)
            max_count = max(class_counts.values())
            
            x_balanced_list, y_balanced_list = [], []
            
            for class_id in range(len(self.class_names)):
                class_mask = (y_flat == class_id)
                x_class = self.x_train[class_mask]
                y_class = self.y_train[class_mask]
                
                if len(x_class) < max_count:
                    num_to_sample = max_count - len(x_class)
                    indices = np.random.choice(len(x_class), num_to_sample, replace=True)
                    x_oversampled = x_class[indices]
                    y_oversampled = y_class[indices]
                    
                    x_balanced_list.append(np.concatenate([x_class, x_oversampled]))
                    y_balanced_list.append(np.concatenate([y_class, y_oversampled]))
                else:
                    x_balanced_list.append(x_class)
                    y_balanced_list.append(y_class)
            
            x_balanced = np.concatenate(x_balanced_list)
            y_balanced = np.concatenate(y_balanced_list)
            
        elif strategy == 'undersample':
            class_counts = Counter(y_flat)
            min_count = min(class_counts.values())
            
            x_balanced_list, y_balanced_list = [], []
            
            for class_id in range(len(self.class_names)):
                class_mask = (y_flat == class_id)
                x_class = self.x_train[class_mask]
                y_class = self.y_train[class_mask]
                
                indices = np.random.choice(len(x_class), min_count, replace=False)
                x_balanced_list.append(x_class[indices])
                y_balanced_list.append(y_class[indices])
            
            x_balanced = np.concatenate(x_balanced_list)
            y_balanced = np.concatenate(y_balanced_list)
        
        else:
            x_balanced, y_balanced = self.x_train, self.y_train
        
        indices = np.random.permutation(len(x_balanced))
        return x_balanced[indices], y_balanced[indices]
    
    def normalize_data(self, mode: str = 'standard') -> None:
        """数据标准化"""
        if mode == 'standard':
            self.x_train = self.x_train.astype('float32') / 255.0
            self.x_val = self.x_val.astype('float32') / 255.0
            self.x_test = self.x_test.astype('float32') / 255.0
        elif mode == 'mean_std':
            mean = np.mean(self.x_train, axis=(0, 1, 2))
            std = np.std(self.x_train, axis=(0, 1, 2))
            self.x_train = (self.x_train.astype('float32') - mean) / (std + 1e-7)
            self.x_val = (self.x_val.astype('float32') - mean) / (std + 1e-7)
            self.x_test = (self.x_test.astype('float32') - mean) / (std + 1e-7)
    
    def create_data_generators(self, batch_size: int = 32) -> Tuple:
        """创建数据生成器"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for data generators")
        
        y_train_cat = to_categorical(self.y_train, len(self.class_names))
        y_val_cat = to_categorical(self.y_val, len(self.class_names))
        
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator()
        
        train_gen = train_datagen.flow(
            self.x_train, y_train_cat,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_gen = val_datagen.flow(
            self.x_val, y_val_cat,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_gen, val_gen, train_datagen, val_datagen
    
    def save_augmented_data(self, output_path: str = '../data/augmented') -> None:
        """保存增强数据"""
        os.makedirs(output_path, exist_ok=True)
        
        np.save(os.path.join(output_path, 'x_train_aug.npy'), self.x_train)
        np.save(os.path.join(output_path, 'y_train_aug.npy'), self.y_train)
        np.save(os.path.join(output_path, 'x_val_aug.npy'), self.x_val)
        np.save(os.path.join(output_path, 'y_val_aug.npy'), self.y_val)
        np.save(os.path.join(output_path, 'x_test_aug.npy'), self.x_test)
        np.save(os.path.join(output_path, 'y_test_aug.npy'), self.y_test)
        
        with open(os.path.join(output_path, 'class_names.txt'), 'w', encoding='utf-8') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        
        print(f"Augmented data saved to {output_path}")
    
    def generate_summary(self, batch_size: int) -> None:
        """生成总结报告"""
        print("=" * 60)
        print("DATA AUGMENTATION SUMMARY")
        print("=" * 60)
        print(f"Training samples: {len(self.x_train)}")
        print(f"Validation samples: {len(self.x_val)}")
        print(f"Test samples: {len(self.x_test)}")
        print(f"Batch size: {batch_size}")
        print(f"Input shape: {self.x_train.shape[1:]}")
        print("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CIFAR-10 Data Augmentation')
    parser.add_argument('--input-dir', default='../data/processed', help='Input data directory')
    parser.add_argument('--output-dir', default='../data/augmented', help='Output data directory')
    parser.add_argument('--balance', choices=['oversample', 'undersample', 'none'], 
                       default='oversample', help='Class balancing strategy')
    parser.add_argument('--normalize', choices=['standard', 'mean_std', 'none'], 
                       default='standard', help='Normalization method')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--visualize', action='store_true', help='Visualize augmentations')
    args = parser.parse_args()
    
    try:
        # 初始化处理器
        augmentor = DataAugmentor(args.input_dir)
        augmentor.load_data()
        
        # 分析原始分布
        augmentor.analyze_class_distribution(augmentor.y_train, "Original Training")
        
        # 可视化增强
        if args.visualize:
            augmentor.visualize_augmentations('base')
            augmentor.visualize_augmentations('strong')
        
        # 类别平衡
        if args.balance != 'none':
            print(f"\nApplying {args.balance} balancing...")
            augmentor.x_train, augmentor.y_train = augmentor.balance_classes(args.balance)
            augmentor.analyze_class_distribution(augmentor.y_train, "Balanced Training")
        
        # 数据标准化
        if args.normalize != 'none':
            print(f"\nApplying {args.normalize} normalization...")
            augmentor.normalize_data(args.normalize)
            print(f"Normalized range: [{augmentor.x_train.min():.3f}, {augmentor.x_train.max():.3f}]")
        
        # 创建数据生成器
        if TF_AVAILABLE:
            train_gen, val_gen, _, _ = augmentor.create_data_generators(args.batch_size)
            print(f"\nTraining batches: {len(train_gen)}")
            print(f"Validation batches: {len(val_gen)}")
        
        # 保存数据
        augmentor.save_augmented_data(args.output_dir)
        
        # 生成总结
        augmentor.generate_summary(args.batch_size)
        
        print("Data augmentation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()