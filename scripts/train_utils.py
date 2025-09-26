#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Script for DASC7606 Assignment I
Training Loops, Optimization, and Hyperparameter Tuning
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import time
from datetime import datetime
import argparse

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import models, callbacks
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.utils import to_categorical
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

class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self, data_path: str = '../data/augmented', 
                 model_path: str = '../models',
                 results_path: str = '../results'):
        self.data_path = data_path
        self.model_path = model_path
        self.results_path = results_path
        self.class_names = []
        self.num_classes = 0
        self.input_shape = None
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.models = {}
        self.training_results = {}
        
    def load_data(self) -> None:
        """加载训练数据"""
        print("Loading training data...")
        
        try:
            self.x_train = np.load(os.path.join(self.data_path, 'x_train_aug.npy'))
            self.y_train = np.load(os.path.join(self.data_path, 'y_train_aug.npy'))
            self.x_val = np.load(os.path.join(self.data_path, 'x_val_aug.npy'))
            self.y_val = np.load(os.path.join(self.data_path, 'y_val_aug.npy'))
            
            with open(os.path.join(self.data_path, 'class_names.txt'), 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            self.num_classes = len(self.class_names)
            self.input_shape = self.x_train.shape[1:]
            
            # 转换为分类格式
            self.y_train = to_categorical(self.y_train, self.num_classes)
            self.y_val = to_categorical(self.y_val, self.num_classes)
            
            print(f"Training data: {self.x_train.shape} -> {self.y_train.shape}")
            print(f"Validation data: {self.x_val.shape} -> {self.y_val.shape}")
            print(f"Number of classes: {self.num_classes}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Please run data augmentation script first: {e}")
    
    def load_models(self, model_names: List[str]) -> None:
        """加载模型架构"""
        print("Loading model architectures...")
        
        for model_name in model_names:
            try:
                config_path = os.path.join(self.model_path, f'{model_name}_config.json')
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if 'class_name' in config:  # Functional API
                    model = models.Model.from_config(config)
                else:  # Sequential API
                    model = models.Sequential.from_config(config)
                
                self.models[model_name] = model
                print(f"Loaded {model_name} architecture")
                
            except FileNotFoundError:
                print(f"Model {model_name} not found, skipping...")
        
        print(f"Loaded {len(self.models)} models for training")
    
    def create_training_configurations(self) -> List[Dict[str, Any]]:
        """创建训练配置"""
        configurations = [
            {
                'name': 'adam_fast',
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 50,
                'patience': 10
            },
            {
                'name': 'adam_slow', 
                'optimizer': 'adam',
                'learning_rate': 0.0001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 15
            },
            {
                'name': 'sgd_momentum',
                'optimizer': 'sgd',
                'learning_rate': 0.01,
                'batch_size': 64,
                'epochs': 80,
                'patience': 12
            },
            {
                'name': 'rmsprop',
                'optimizer': 'rmsprop', 
                'learning_rate': 0.001,
                'batch_size': 128,
                'epochs': 60,
                'patience': 10
            }
        ]
        
        return configurations
    
    def create_callbacks(self, patience: int, model_name: str, config_name: str) -> List:
        """创建训练回调函数"""
        if not TF_AVAILABLE:
            return []
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_path, f'{model_name}_{config_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.CSVLogger(
                filename=os.path.join(self.results_path, f'{model_name}_{config_name}_log.csv')
            )
        ]
        
        # 只在有TensorBoard时添加
        try:
            tensorboard_callback = callbacks.TensorBoard(
                log_dir=os.path.join('../logs', f'{model_name}_{config_name}_{timestamp}'),
                histogram_freq=1
            )
            callbacks_list.append(tensorboard_callback)
        except:
            pass
        
        return callbacks_list
    
    def compile_model(self, model: models.Model, optimizer: str, learning_rate: float) -> models.Model:
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
    
    def train_model(self, model: models.Model, model_name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """训练单个模型"""
        if not TF_AVAILABLE:
            return None
        
        print(f"\nTraining {model_name} with {config['name']} configuration...")
        print("-" * 50)
        
        # 创建模型副本
        model_copy = models.clone_model(model)
        model_copy.build(model.input_shape)
        model_copy = self.compile_model(model_copy, config['optimizer'], config['learning_rate'])
        
        # 创建回调
        callbacks_list = self.create_callbacks(config['patience'], model_name, config['name'])
        
        # 训练模型
        start_time = time.time()
        
        try:
            history = model_copy.fit(
                self.x_train, self.y_train,
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                validation_data=(self.x_val, self.y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            results = {
                'model_name': model_name,
                'config_name': config['name'],
                'history': history.history,
                'training_time': training_time,
                'final_epoch': len(history.history['loss']),
                'best_val_accuracy': max(history.history['val_accuracy']),
                'best_val_loss': min(history.history['val_loss']),
                'model': model_copy
            }
            
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error training {model_name} with {config['name']}: {e}")
            return None
    
    def train_all_models(self, training_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """训练所有模型"""
        print("Starting model training...")
        
        training_results = {}
        
        for model_name, model in self.models.items():
            model_results = {}
            
            for config in training_configs:
                results = self.train_model(model, model_name, config)
                if results:
                    model_results[config['name']] = results
            
            training_results[model_name] = model_results
        
        self.training_results = training_results
        print(f"\nTraining completed for {len(training_results)} models")
        return training_results
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                             model_name: str, config_name: str, 
                             save_plots: bool = False) -> None:
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{model_name} - {config_name}\nLoss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title(f'{model_name} - {config_name}\nAccuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(self.results_path, exist_ok=True)
            plot_path = os.path.join(self.results_path, f'{model_name}_{config_name}_training.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {plot_path}")
        
        plt.show()
    
    def analyze_training_results(self, save_plots: bool = False) -> pd.DataFrame:
        """分析训练结果"""
        if not self.training_results:
            print("No training results to analyze")
            return pd.DataFrame()
        
        results_data = []
        
        for model_name, model_results in self.training_results.items():
            for config_name, results in model_results.items():
                results_data.append({
                    'Model': model_name,
                    'Config': config_name,
                    'Best Val Accuracy': results['best_val_accuracy'],
                    'Best Val Loss': results['best_val_loss'],
                    'Training Time (s)': results['training_time'],
                    'Epochs': results['final_epoch']
                })
                
                # 绘制训练历史
                self.plot_training_history(results['history'], model_name, config_name, save_plots)
        
        df = pd.DataFrame(results_data)
        
        # 显示结果表格
        print("\nTraining Results Comparison:")
        print("=" * 80)
        print(df.round(4))
        
        # 绘制比较图
        if not df.empty:
            self.plot_comparison_charts(df, save_plots)
        
        return df
    
    def plot_comparison_charts(self, df: pd.DataFrame, save_plots: bool = False) -> None:
        """绘制比较图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 验证准确率比较
        sns.barplot(data=df, x='Model', y='Best Val Accuracy', hue='Config', ax=ax1)
        ax1.set_title('Validation Accuracy by Model and Config')
        ax1.tick_params(axis='x', rotation=45)
        
        # 训练时间比较
        sns.barplot(data=df, x='Model', y='Training Time (s)', hue='Config', ax=ax2)
        ax2.set_title('Training Time by Model and Config')
        ax2.tick_params(axis='x', rotation=45)
        
        # 验证损失比较
        sns.barplot(data=df, x='Model', y='Best Val Loss', hue='Config', ax=ax3)
        ax3.set_title('Validation Loss by Model and Config')
        ax3.tick_params(axis='x', rotation=45)
        
        # 训练轮数比较
        sns.barplot(data=df, x='Model', y='Epochs', hue='Config', ax=ax4)
        ax4.set_title('Training Epochs by Model and Config')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.results_path, 'training_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {plot_path}")
        
        plt.show()
    
    def hyperparameter_tuning(self, model_name: str, 
                            param_grid: Dict[str, List] = None) -> pd.DataFrame:
        """超参数调优"""
        if not TF_AVAILABLE or model_name not in self.models:
            return pd.DataFrame()
        
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.001, 0.0005, 0.0001],
                'batch_size': [32, 64, 128],
                'optimizer': ['adam', 'sgd']
            }
        
        model = self.models[model_name]
        results = []
        
        print(f"\nPerforming hyperparameter tuning on {model_name}...")
        
        for lr in param_grid['learning_rate']:
            for bs in param_grid['batch_size']:
                for opt in param_grid['optimizer']:
                    print(f"Tuning: lr={lr}, bs={bs}, opt={opt}")
                    
                    # 创建模型副本
                    model_copy = models.clone_model(model)
                    model_copy.build(model.input_shape)
                    model_copy = self.compile_model(model_copy, opt, lr)
                    
                    # 简短训练进行测试
                    history = model_copy.fit(
                        self.x_train, self.y_train,
                        batch_size=bs,
                        epochs=10,  # 简短的epochs用于调优
                        validation_data=(self.x_val, self.y_val),
                        verbose=0
                    )
                    
                    best_val_acc = max(history.history['val_accuracy'])
                    best_val_loss = min(history.history['val_loss'])
                    
                    results.append({
                        'learning_rate': lr,
                        'batch_size': bs,
                        'optimizer': opt,
                        'val_accuracy': best_val_acc,
                        'val_loss': best_val_loss
                    })
        
        tuning_df = pd.DataFrame(results)
        
        # 显示最佳参数
        best_params = tuning_df.loc[tuning_df['val_accuracy'].idxmax()]
        print("\nBest hyperparameters:")
        print(best_params)
        
        return tuning_df
    
    def evaluate_models(self) -> pd.DataFrame:
        """在验证集上评估模型"""
        if not TF_AVAILABLE:
            return pd.DataFrame()
        
        evaluation_results = []
        
        for model_name, model_results in self.training_results.items():
            for config_name, results in model_results.items():
                model_path = os.path.join(self.model_path, f'{model_name}_{config_name}_best.h5')
                
                try:
                    model = models.load_model(model_path)
                    
                    # 评估模型
                    val_loss, val_accuracy = model.evaluate(self.x_val, self.y_val, verbose=0)
                    
                    evaluation_results.append({
                        'Model': model_name,
                        'Config': config_name,
                        'Validation Loss': val_loss,
                        'Validation Accuracy': val_accuracy,
                        'Training Time (s)': results['training_time']
                    })
                    
                    print(f"Evaluated {model_name}_{config_name}: {val_accuracy:.4f}")
                    
                except Exception as e:
                    print(f"Error evaluating {model_name}_{config_name}: {e}")
                    continue
        
        eval_df = pd.DataFrame(evaluation_results)
        
        if not eval_df.empty:
            # 显示评估结果
            print("\nValidation Set Evaluation Results:")
            print("=" * 60)
            print(eval_df.round(4))
            
            # 找到最佳模型
            best_model_row = eval_df.loc[eval_df['Validation Accuracy'].idxmax()]
            print(f"\nBest Model: {best_model_row['Model']} with {best_model_row['Config']}")
            print(f"Best Validation Accuracy: {best_model_row['Validation Accuracy']:.4f}")
        
        return eval_df
    
    def save_results(self, results_df: pd.DataFrame, eval_df: pd.DataFrame, 
                    tuning_df: pd.DataFrame = None) -> None:
        """保存训练结果"""
        os.makedirs(self.results_path, exist_ok=True)
        
        # 保存CSV结果
        results_df.to_csv(os.path.join(self.results_path, 'training_results.csv'), index=False)
        eval_df.to_csv(os.path.join(self.results_path, 'validation_results.csv'), index=False)
        
        if tuning_df is not None:
            tuning_df.to_csv(os.path.join(self.results_path, 'hyperparameter_tuning.csv'), index=False)
        
        # 保存详细的训练历史
        training_history = {}
        for model_name, model_results in self.training_results.items():
            training_history[model_name] = {}
            for config_name, results in model_results.items():
                training_history[model_name][config_name] = {
                    'history': {k: [float(x) for x in v] for k, v in results['history'].items()},
                    'training_time': results['training_time'],
                    'best_metrics': {
                        'val_accuracy': float(results['best_val_accuracy']),
                        'val_loss': float(results['best_val_loss'])
                    }
                }
        
        with open(os.path.join(self.results_path, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # 保存最佳模型信息
        if not eval_df.empty:
            best_model_info = eval_df.loc[eval_df['Validation Accuracy'].idxmax()].to_dict()
            with open(os.path.join(self.results_path, 'best_model_info.json'), 'w') as f:
                json.dump(best_model_info, f, indent=2)
        
        print(f"All results saved to {self.results_path}")
    
    def generate_summary(self, eval_df: pd.DataFrame) -> None:
        """生成训练总结"""
        print("=" * 60)
        print("MODEL TRAINING SUMMARY REPORT")
        print("=" * 60)
        
        total_runs = sum(len(results) for results in self.training_results.values())
        print(f"Total training runs: {total_runs}")
        print(f"Models trained: {len(self.training_results)}")
        
        if not eval_df.empty:
            best_model = eval_df.loc[eval_df['Validation Accuracy'].idxmax()]
            print(f"\nBest Performing Model:")
            print(f"  Model: {best_model['Model']}")
            print(f"  Config: {best_model['Config']}")
            print(f"  Validation Accuracy: {best_model['Validation Accuracy']:.4f}")
            print(f"  Training Time: {best_model['Training Time (s)']:.2f}s")
            
            # 训练统计
            avg_accuracy = eval_df['Validation Accuracy'].mean()
            max_accuracy = eval_df['Validation Accuracy'].max()
            min_accuracy = eval_df['Validation Accuracy'].min()
            
            print(f"\nPerformance Statistics:")
            print(f"  Average Accuracy: {avg_accuracy:.4f}")
            print(f"  Maximum Accuracy: {max_accuracy:.4f}")
            print(f"  Minimum Accuracy: {min_accuracy:.4f}")
            print(f"  Accuracy Range: {max_accuracy - min_accuracy:.4f}")
        
        print(f"\nResults saved to: {self.results_path}")
        print("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data-dir', default='../data/augmented', help='Input data directory')
    parser.add_argument('--model-dir', default='../models', help='Model directory')
    parser.add_argument('--results-dir', default='../results', help='Results directory')
    parser.add_argument('--models', nargs='+', default=['simple_cnn', 'advanced_cnn', 'lightweight_cnn'],
                       help='Models to train')
    parser.add_argument('--hyperparameter-tuning', action='store_true', 
                       help='Perform hyperparameter tuning')
    parser.add_argument('--save-plots', action='store_true', help='Save visualization plots')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum training epochs')
    args = parser.parse_args()
    
    try:
        # 初始化训练器
        trainer = ModelTrainer(args.data_dir, args.model_dir, args.results_dir)
        
        # 加载数据
        trainer.load_data()
        
        # 加载模型
        trainer.load_models(args.models)
        
        if not trainer.models:
            print("No models available for training")
            return
        
        # 创建训练配置
        training_configs = trainer.create_training_configurations()
        
        # 调整epochs
        for config in training_configs:
            config['epochs'] = min(config['epochs'], args.epochs)
        
        # 训练模型
        training_results = trainer.train_all_models(training_configs)
        
        if not training_results:
            print("No models were successfully trained")
            return
        
        # 分析结果
        results_df = trainer.analyze_training_results(args.save_plots)
        
        # 超参数调优
        tuning_df = None
        if args.hyperparameter_tuning and trainer.models:
            first_model = list(trainer.models.keys())[0]
            tuning_df = trainer.hyperparameter_tuning(first_model)
        
        # 评估模型
        eval_df = trainer.evaluate_models()
        
        # 保存结果
        trainer.save_results(results_df, eval_df, tuning_df)
        
        # 生成总结
        trainer.generate_summary(eval_df)
        
        print("Model training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()