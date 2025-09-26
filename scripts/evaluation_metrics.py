#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation Script for DASC7606 Assignment I
Performance Metrics, Visualization, and Error Analysis
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
import argparse
from datetime import datetime

# 机器学习指标
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, roc_curve, auc, 
                           precision_score, recall_score, f1_score, 
                           accuracy_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import models
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

# 设置
plt.style.use('ggplot')
sns.set_palette("husl")
np.random.seed(42)

class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self, data_path: str = '../data/augmented', 
                 model_path: str = '../models',
                 results_path: str = '../results'):
        self.data_path = data_path
        self.model_path = model_path
        self.results_path = results_path
        self.class_names = []
        self.num_classes = 0
        self.x_test, self.y_test, self.y_test_cat = None, None, None
        self.models_evaluation = {}
        
    def load_data(self) -> None:
        """加载评估数据"""
        print("Loading evaluation data...")
        
        try:
            self.x_test = np.load(os.path.join(self.data_path, 'x_test_aug.npy'))
            self.y_test = np.load(os.path.join(self.data_path, 'y_test_aug.npy'))
            
            with open(os.path.join(self.data_path, 'class_names.txt'), 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            self.num_classes = len(self.class_names)
            self.y_test_cat = to_categorical(self.y_test, self.num_classes)
            
            print(f"Test data: {self.x_test.shape} -> {self.y_test_cat.shape}")
            print(f"Number of classes: {self.num_classes}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Please run data augmentation script first: {e}")
    
    def load_model_predictions(self, model_name: str) -> Optional[Dict[str, Any]]:
        """加载模型并进行预测"""
        if not TF_AVAILABLE:
            return None
            
        model_file = os.path.join(self.model_path, f"{model_name}_best.h5")
        
        try:
            model = models.load_model(model_file)
            
            # 进行预测
            y_pred_proba = model.predict(self.x_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # 计算置信度
            confidence_scores = np.max(y_pred_proba, axis=1)
            
            predictions = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confidence_scores': confidence_scores,
                'model_name': model_name
            }
            
            print(f"Predictions completed for {model_name}")
            return predictions
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """评估模型性能"""
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # 分类报告
        clf_report = classification_report(y_true, y_pred, 
                                         target_names=self.class_names, 
                                         output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC曲线计算
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 微平均和宏平均
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.num_classes
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': clf_report,
            'confusion_matrix': cm,
            'roc_curves': {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc},
            'per_class_metrics': {
                'precision': precision_score(y_true, y_pred, average=None),
                'recall': recall_score(y_true, y_pred, average=None),
                'f1': f1_score(y_true, y_pred, average=None)
            }
        }
        
        return results
    
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      confidence_scores: np.ndarray) -> Dict[str, Any]:
        """分析模型错误"""
        errors_mask = (y_true != y_pred)
        
        error_analysis = {
            'total_errors': np.sum(errors_mask),
            'error_rate': np.mean(errors_mask),
            'error_indices': np.where(errors_mask)[0],
            'correct_indices': np.where(~errors_mask)[0],
            'error_confidences': confidence_scores[errors_mask],
            'correct_confidences': confidence_scores[~errors_mask],
            'misclassification_pairs': []
        }
        
        # 分析错误类型
        error_pairs = {}
        for true_label, pred_label in zip(y_true[errors_mask], y_pred[errors_mask]):
            pair = (true_label, pred_label)
            error_pairs[pair] = error_pairs.get(pair, 0) + 1
        
        for (true_idx, pred_idx), count in error_pairs.items():
            error_analysis['misclassification_pairs'].append({
                'true_class': self.class_names[true_idx],
                'predicted_class': self.class_names[pred_idx],
                'count': count,
                'percentage': count / len(error_analysis['error_indices']) * 100
            })
        
        error_analysis['misclassification_pairs'].sort(key=lambda x: x['count'], reverse=True)
        return error_analysis
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, 
                             normalize: bool = True, save_plot: bool = False) -> None:
        """绘制混淆矩阵"""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'{model_name} - Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = f'{model_name} - Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_path, f'{model_name}_confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {plot_path}")
        
        plt.show()
    
    def plot_roc_curves(self, roc_data: Dict[str, Any], model_name: str, 
                       save_plot: bool = False) -> None:
        """绘制ROC曲线"""
        fpr, tpr, roc_auc = roc_data['fpr'], roc_data['tpr'], roc_data['roc_auc']
        
        plt.figure(figsize=(10, 8))
        
        # 每个类别的ROC曲线
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        for i, color in zip(range(self.num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # 平均ROC曲线
        plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='navy', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_path, f'{model_name}_roc_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {plot_path}")
        
        plt.show()
    
    def plot_error_analysis(self, error_analysis: Dict[str, Any], model_name: str,
                          save_plot: bool = False) -> None:
        """绘制错误分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 置信度分布
        ax1.hist(error_analysis['correct_confidences'], bins=30, alpha=0.7, 
                label='Correct Predictions', color='green')
        ax1.hist(error_analysis['error_confidences'], bins=30, alpha=0.7, 
                label='Wrong Predictions', color='red')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 最常见的错误类型
        top_errors = error_analysis['misclassification_pairs'][:10]
        error_labels = [f"{err['true_class']}→{err['predicted_class']}" for err in top_errors]
        error_counts = [err['count'] for err in top_errors]
        
        ax2.barh(error_labels, error_counts, color='coral')
        ax2.set_xlabel('Error Count')
        ax2.set_title('Top Misclassification Patterns')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'{model_name} - Error Analysis')
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_path, f'{model_name}_error_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Error analysis saved to {plot_path}")
        
        plt.show()
        
        # 打印错误统计
        print(f"Total errors: {error_analysis['total_errors']}")
        print(f"Error rate: {error_analysis['error_rate']:.3f}")
        print(f"Avg confidence (correct): {np.mean(error_analysis['correct_confidences']):.3f}")
        print(f"Avg confidence (errors): {np.mean(error_analysis['error_confidences']):.3f}")
        
        print("\nTop misclassification patterns:")
        for i, error in enumerate(top_errors, 1):
            print(f"{i:2d}. {error['true_class']:>12} → {error['predicted_class']:<12} "
                  f"({error['count']:3d} times, {error['percentage']:5.1f}%)")
    
    def evaluate_single_model(self, model_name: str, save_plots: bool = False) -> Optional[Dict[str, Any]]:
        """评估单个模型"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # 加载模型预测
        predictions = self.load_model_predictions(model_name)
        if predictions is None:
            return None
        
        # 评估性能
        performance = self.evaluate_model_performance(self.y_test, predictions['y_pred'], 
                                                    predictions['y_pred_proba'])
        
        # 错误分析
        error_analysis = self.analyze_errors(self.y_test, predictions['y_pred'], 
                                           predictions['confidence_scores'])
        
        # 可视化
        self.plot_confusion_matrix(performance['confusion_matrix'], model_name, save_plot=save_plots)
        self.plot_roc_curves(performance['roc_curves'], model_name, save_plot=save_plots)
        self.plot_error_analysis(error_analysis, model_name, save_plot=save_plots)
        
        # 打印分类报告
        print("\nClassification Report:")
        print("-" * 50)
        print(classification_report(self.y_test, predictions['y_pred'], 
                                  target_names=self.class_names))
        
        # 合并结果
        evaluation_results = {
            'model_name': model_name,
            'predictions': predictions,
            'performance': performance,
            'error_analysis': error_analysis
        }
        
        return evaluation_results
    
    def compare_models(self, save_plots: bool = False) -> pd.DataFrame:
        """比较多个模型"""
        if not self.models_evaluation:
            print("No models to compare")
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, evaluation in self.models_evaluation.items():
            perf = evaluation['performance']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': perf['accuracy'],
                'Precision': perf['precision'],
                'Recall': perf['recall'],
                'F1-Score': perf['f1_score'],
                'Micro AUC': perf['roc_curves']['roc_auc']['micro'],
                'Macro AUC': perf['roc_curves']['roc_auc']['macro'],
                'Error Rate': evaluation['error_analysis']['error_rate']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 绘制比较图
        if save_plots:
            self.plot_model_comparison(df)
        
        return df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """绘制模型比较图"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Micro AUC', 'Macro AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            bars = axes[i].bar(comparison_df['Model'], comparison_df[metric],
                             color=plt.cm.Set1(range(len(comparison_df))))
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1.1)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, comparison_df[metric]):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_path, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {plot_path}")
        
        plt.show()
    
    def save_results(self, comparison_df: pd.DataFrame) -> None:
        """保存评估结果"""
        os.makedirs(self.results_path, exist_ok=True)
        
        # 保存模型比较结果
        comparison_df.to_csv(os.path.join(self.results_path, 'model_comparison.csv'), index=False)
        
        # 保存详细结果
        evaluation_summary = {}
        for model_name, evaluation in self.models_evaluation.items():
            evaluation_summary[model_name] = {
                'accuracy': evaluation['performance']['accuracy'],
                'precision': evaluation['performance']['precision'],
                'recall': evaluation['performance']['recall'],
                'f1_score': evaluation['performance']['f1_score'],
                'micro_auc': evaluation['performance']['roc_curves']['roc_auc']['micro'],
                'macro_auc': evaluation['performance']['roc_curves']['roc_auc']['macro'],
                'error_rate': evaluation['error_analysis']['error_rate'],
                'total_errors': evaluation['error_analysis']['total_errors'],
                'top_misclassifications': evaluation['error_analysis']['misclassification_pairs'][:5]
            }
        
        with open(os.path.join(self.results_path, 'evaluation_summary.json'), 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        # 保存分类报告
        for model_name, evaluation in self.models_evaluation.items():
            clf_report = evaluation['performance']['classification_report']
            report_df = pd.DataFrame(clf_report).transpose()
            report_df.to_csv(os.path.join(self.results_path, f'{model_name}_classification_report.csv'))
        
        print(f"Evaluation results saved to {self.results_path}")
    
    def generate_summary(self, comparison_df: pd.DataFrame) -> None:
        """生成总结报告"""
        print("=" * 70)
        print("FINAL MODEL EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"Models evaluated: {len(self.models_evaluation)}")
        print(f"Test samples: {len(self.x_test)}")
        print(f"Number of classes: {self.num_classes}")
        
        if not comparison_df.empty:
            best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            
            print(f"\n🏆 BEST MODEL: {best_model['Model']}")
            print(f"   Accuracy: {best_model['Accuracy']:.4f}")
            print(f"   F1-Score: {best_model['F1-Score']:.4f}")
            print(f"   Macro AUC: {best_model['Macro AUC']:.4f}")
            
            print(f"\n📊 Performance Range:")
            print(f"   Accuracy: {comparison_df['Accuracy'].min():.4f} - {comparison_df['Accuracy'].max():.4f}")
            
            print(f"\n🔍 Error Analysis:")
            for model_name, evaluation in self.models_evaluation.items():
                error_rate = evaluation['error_analysis']['error_rate']
                top_error = evaluation['error_analysis']['misclassification_pairs'][0]
                print(f"   {model_name}: Error rate {error_rate:.3f}")
        
        print(f"\n💾 Results saved to: {self.results_path}")
        print("=" * 70)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--data-dir', default='../data/augmented', help='Input data directory')
    parser.add_argument('--model-dir', default='../models', help='Model directory')
    parser.add_argument('--results-dir', default='../results', help='Results directory')
    parser.add_argument('--models', nargs='+', 
                       default=['simple_cnn', 'advanced_cnn', 'lightweight_cnn'],
                       help='Models to evaluate')
    parser.add_argument('--save-plots', action='store_true', help='Save visualization plots')
    parser.add_argument('--compare-only', action='store_true', 
                       help='Only compare existing results')
    args = parser.parse_args()
    
    try:
        # 初始化评估器
        evaluator = ModelEvaluator(args.data_dir, args.model_dir, args.results_dir)
        
        if not args.compare_only:
            # 加载数据
            evaluator.load_data()
            
            # 评估模型
            for model_name in args.models:
                evaluation = evaluator.evaluate_single_model(model_name, args.save_plots)
                if evaluation:
                    evaluator.models_evaluation[model_name] = evaluation
            
            if not evaluator.models_evaluation:
                print("No models were successfully evaluated")
                return
            
            # 比较模型
            comparison_df = evaluator.compare_models(args.save_plots)
            
            # 保存结果
            evaluator.save_results(comparison_df)
        else:
            # 仅比较现有结果
            comparison_file = os.path.join(args.results_dir, 'model_comparison.csv')
            if os.path.exists(comparison_file):
                comparison_df = pd.read_csv(comparison_file)
                evaluator.plot_model_comparison(comparison_df)
            else:
                print("No existing results found for comparison")
                return
        
        # 生成总结
        if not args.compare_only:
            evaluator.generate_summary(comparison_df)
        
        print("Model evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()