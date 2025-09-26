#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Building Script for DASC7606 Assignment I
CNN Architectures and Transfer Learning
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import argparse

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, Model
    from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import plot_model, to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

# 设置
plt.style.use('ggplot')
sns.set_palette("husl")
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)

class ModelBuilder:
    """模型构建器类"""
    
    def __init__(self, data_path: str = '../data/augmented'):
        self.data_path = data_path
        self.class_names = []
        self.num_classes = 0
        self.input_shape = None
        self.models = {}
        
    def load_data(self) -> None:
        """加载数据"""
        print("Loading augmented data...")
        
        try:
            x_train = np.load(os.path.join(self.data_path, 'x_train_aug.npy'))
            y_train = np.load(os.path.join(self.data_path, 'y_train_aug.npy'))
            
            with open(os.path.join(self.data_path, 'class_names.txt'), 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            self.num_classes = len(self.class_names)
            self.input_shape = x_train.shape[1:]
            
            print(f"Data loaded: input_shape={self.input_shape}, classes={self.num_classes}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Please run data augmentation script first: {e}")
    
    def create_simple_cnn(self) -> Model:
        """创建简单CNN"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_advanced_cnn(self) -> Model:
        """创建高级CNN"""
        inputs = layers.Input(shape=self.input_shape)
        
        # 卷积块
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # 分类器
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_transfer_learning_model(self, base_model_name: str = 'VGG16', 
                                     trainable_layers: int = 5) -> Model:
        """创建迁移学习模型"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for transfer learning")
        
        # 选择基础模型
        if base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, 
                              input_shape=self.input_shape)
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                 input_shape=self.input_shape)
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                       input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # 冻结层
        base_model.trainable = False
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        
        # 添加分类头
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model, base_model
    
    def create_lightweight_cnn(self) -> Model:
        """创建轻量级CNN"""
        model = models.Sequential([
            layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same',
                                  input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model: Model, optimizer: str = 'adam', 
                     learning_rate: float = 0.001) -> Model:
        """编译模型"""
        if not TF_AVAILABLE:
            return model
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            opt = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def analyze_complexity(self, model: Model) -> Dict[str, Any]:
        """分析模型复杂度"""
        if not TF_AVAILABLE:
            return {}
        
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        return {
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params': total_params,
            'num_layers': len(model.layers),
            'model_size_mb': (total_params * 4) / (1024 * 1024),
            'trainable_percentage': (trainable_params / total_params) * 100
        }
    
    def save_model_architecture(self, model: Model, model_name: str, 
                               model_dir: str = '../models') -> None:
        """保存模型架构"""
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存配置
        config = model.get_config()
        with open(os.path.join(model_dir, f'{model_name}_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # 保存摘要
        with open(os.path.join(model_dir, f'{model_name}_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        print(f"Saved {model_name} architecture")
    
    def build_all_models(self) -> Dict[str, Model]:
        """构建所有模型"""
        print("Building all models...")
        
        self.models['simple_cnn'] = self.create_simple_cnn()
        self.models['advanced_cnn'] = self.create_advanced_cnn()
        self.models['lightweight_cnn'] = self.create_lightweight_cnn()
        
        # 迁移学习模型
        try:
            vgg_model, _ = self.create_transfer_learning_model('VGG16')
            self.models['vgg16_transfer'] = vgg_model
        except Exception as e:
            print(f"Could not create VGG16 model: {e}")
        
        print(f"Built {len(self.models)} models")
        return self.models
    
    def compare_complexities(self) -> None:
        """比较模型复杂度"""
        if not self.models:
            print("No models to compare")
            return
        
        print("\nModel Complexity Comparison:")
        print("=" * 80)
        print(f"{'Model':<20} {'Trainable':<12} {'Total':<12} {'Layers':<8} {'Size (MB)':<10}")
        print("-" * 80)
        
        for name, model in self.models.items():
            complexity = self.analyze_complexity(model)
            if complexity:
                print(f"{name:<20} {complexity['trainable_params']:<12,} "
                      f"{complexity['total_params']:<12,} {complexity['num_layers']:<8} "
                      f"{complexity['model_size_mb']:<10.2f}")

# ...existing code...

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CNN Model Building')
    parser.add_argument('--data-dir', default='../data/augmented', help='Input data directory')
    parser.add_argument('--model-dir', default='../models', help='Output model directory')
    parser.add_argument('--build-transfer', action='store_true', help='Build transfer learning models')
    args = parser.parse_args()
    
    try:
        # 初始化构建器
        builder = ModelBuilder(args.data_dir)
        builder.load_data()
        
        # 构建模型
        models = builder.build_all_models()
        
        # 比较复杂度
        builder.compare_complexities()
        
        # 保存模型架构
        for name, model in models.items():
            builder.save_model_architecture(model, name, args.model_dir)
        
        # 生成总结报告
        print("\n" + "="*60)
        print("MODEL BUILDING SUMMARY")
        print("="*60)
        print(f"Input shape: {builder.input_shape}")
        print(f"Number of classes: {builder.num_classes}")
        print(f"Models built: {len(models)}")
        
        if TF_AVAILABLE:
            print(f"\nModel details:")
            for name, model in models.items():
                complexity = builder.analyze_complexity(model)
                if complexity:
                    print(f"  {name}: {complexity['total_params']:,} parameters, "
                          f"{complexity['num_layers']} layers")
        
        print(f"\nModels saved to: {args.model_dir}")
        print("="*60)
        
        print("Model building completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()