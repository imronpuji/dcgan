import torch
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to fix authorization error
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import warnings
from itertools import cycle

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_roc_curves(true_labels, pred_probs, class_names, save_dir):
    os.makedirs(f'{save_dir}/roc_curves', exist_ok=True)
    
    # Get actual number of classes from the data
    unique_classes = np.unique(true_labels)
    n_classes = len(unique_classes)
    
    # Create one-hot encoded labels
    y_test = np.zeros((len(true_labels), n_classes))
    for i, label in enumerate(true_labels):
        class_idx = np.where(unique_classes == label)[0][0]
        y_test[i, class_idx] = 1
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange'])
    
    for i, color in zip(range(n_classes), colors):
        class_idx = unique_classes[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of {class_names[class_idx]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/roc_curves/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def create_evaluation_visualizations(true_labels, pred_labels, class_names, save_dir):
    os.makedirs(f'{save_dir}/confusion_matrix', exist_ok=True)
    os.makedirs(f'{save_dir}/class_metrics', exist_ok=True)
    
    # Get unique classes present in the data
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)
    all_unique_labels = np.union1d(unique_true_labels, unique_pred_labels)
    
    # Print debugging info
    print(f"Unique classes in true_labels: {unique_true_labels}")
    print(f"Unique classes in pred_labels: {unique_pred_labels}")
    print(f"All unique labels: {all_unique_labels}")
    print(f"Class names provided: {class_names}")
    
    # Map class names based on actual labels present
    actual_class_names = [class_names[i] for i in all_unique_labels if i < len(class_names)]
    
    print(f"Using class names: {actual_class_names}")
    print(f"Missing classes from test data: {[class_names[i] for i in range(len(class_names)) if i not in all_unique_labels]}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=all_unique_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=actual_class_names, 
                yticklabels=actual_class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report with proper labels and zero_division handling
    report = classification_report(
        true_labels, pred_labels,
        labels=all_unique_labels,
        target_names=actual_class_names,
        output_dict=True,
        zero_division=0  # Set to 0 to avoid warnings
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f'{save_dir}/class_metrics/classification_report.csv')
    
    # Calculate metrics with proper handling of zero division
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, labels=all_unique_labels, average='weighted', zero_division=0),
        'recall': recall_score(true_labels, pred_labels, labels=all_unique_labels, average='weighted', zero_division=0),
        'f1': f1_score(true_labels, pred_labels, labels=all_unique_labels, average='weighted', zero_division=0)
    }
    
    # Add per-class metrics
    precision_per_class = precision_score(true_labels, pred_labels, labels=all_unique_labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, pred_labels, labels=all_unique_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, pred_labels, labels=all_unique_labels, average=None, zero_division=0)
    
    per_class_metrics = {}
    for i, label in enumerate(all_unique_labels):
        class_name = actual_class_names[i]
        per_class_metrics[f'{class_name}_precision'] = float(precision_per_class[i])
        per_class_metrics[f'{class_name}_recall'] = float(recall_per_class[i])
        per_class_metrics[f'{class_name}_f1'] = float(f1_per_class[i])
    
    metrics.update(per_class_metrics)
    
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"✅ Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"✅ Weighted Precision: {metrics['precision']:.4f}")
    print(f"✅ Weighted Recall: {metrics['recall']:.4f}")
    print(f"✅ Weighted F1-Score: {metrics['f1']:.4f}")
    
    print(f"\n📈 Per-Class Performance:")
    for i, label in enumerate(all_unique_labels):
        class_name = actual_class_names[i]
        print(f"   {class_name:15} -> P: {precision_per_class[i]:.3f}, R: {recall_per_class[i]:.3f}, F1: {f1_per_class[i]:.3f}")
    
    return df_report, metrics
    
def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train', linewidth=2)
    plt.plot(history['val_acc'], label='Validation', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot  
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual plots for better visibility
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train', linewidth=2)
    plt.plot(history['val_acc'], label='Validation', linewidth=2)
    plt.title('Model Accuracy Over Time', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss Over Time', fontsize=16, fontweight='bold')
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()