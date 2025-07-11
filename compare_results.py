#!/usr/bin/env python3
"""
📊 COMPARISON SCRIPT
===================
Script untuk membandingkan hasil sebelum dan sesudah perbaikan
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def load_results():
    """Load results from both runs"""
    results = {}
    
    # Load original results
    original_path = "results/execution_info.json"
    if os.path.exists(original_path):
        with open(original_path, 'r') as f:
            results['original'] = json.load(f)
    else:
        print("❌ Original results not found")
        return None
    
    # Load improved results
    improved_path = "results/improved_execution_info.json"
    if os.path.exists(improved_path):
        with open(improved_path, 'r') as f:
            results['improved'] = json.load(f)
    else:
        print("❌ Improved results not found - run the improved training first")
        return None
    
    return results

def compare_metrics(results):
    """Compare key metrics between runs"""
    original = results['original']
    improved = results['improved']
    
    print("="*60)
    print("📊 METRICS COMPARISON")
    print("="*60)
    
    # Extract metrics
    orig_acc = original['final_metrics']['accuracy']
    impr_acc = improved['final_metrics']['accuracy']
    
    orig_f1 = original['final_metrics']['f1']
    impr_f1 = improved['final_metrics']['f1']
    
    orig_epochs = original.get('num_epochs_completed', original.get('num_epochs', 1))
    impr_epochs = improved.get('num_epochs_completed', improved.get('num_epochs_configured', 25))
    
    print(f"📈 ACCURACY:")
    print(f"  Before: {orig_acc:.4f} ({orig_acc*100:.2f}%)")
    print(f"  After:  {impr_acc:.4f} ({impr_acc*100:.2f}%)")
    print(f"  Change: {(impr_acc-orig_acc)*100:+.2f}%")
    
    print(f"\n📈 F1 SCORE:")
    print(f"  Before: {orig_f1:.4f}")
    print(f"  After:  {impr_f1:.4f}")
    print(f"  Change: {impr_f1-orig_f1:+.4f}")
    
    print(f"\n📊 TRAINING EPOCHS:")
    print(f"  Before: {orig_epochs} epochs")
    print(f"  After:  {impr_epochs} epochs")
    
    print(f"\n⏱️ TRAINING TIME:")
    print(f"  Before: {original['training_time']}")
    print(f"  After:  {improved['training_time']}")
    
    # Configuration changes
    print(f"\n🔧 CONFIGURATION CHANGES:")
    print(f"  Image Size: 200x200 → 224x224")
    print(f"  Early Stopping: 10 → {improved.get('early_stopping_patience', 15)}")
    print(f"  Max Epochs: {original.get('num_epochs', 10)} → {improved.get('num_epochs_configured', 25)}")
    print(f"  Learning Rate Scheduler: Added")
    
    return {
        'accuracy_improvement': (impr_acc - orig_acc) * 100,
        'f1_improvement': impr_f1 - orig_f1,
        'original_accuracy': orig_acc * 100,
        'improved_accuracy': impr_acc * 100
    }

def create_comparison_plot(results, comparison_stats):
    """Create visual comparison"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Accuracy Comparison
    categories = ['Original', 'Improved']
    accuracies = [comparison_stats['original_accuracy'], comparison_stats['improved_accuracy']]
    
    bars1 = ax1.bar(categories, accuracies, color=['lightcoral', 'lightgreen'])
    ax1.set_title('🎯 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(90, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Configuration Comparison
    configs = ['Epochs\n(max)', 'Image Size', 'Early Stop\nPatience']
    original_configs = [
        results['original'].get('num_epochs', 10),
        '200x200',
        10
    ]
    improved_configs = [
        results['improved'].get('num_epochs_configured', 25),
        '224x224', 
        results['improved'].get('early_stopping_patience', 15)
    ]
    
    x = range(len(configs))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], [original_configs[0], 200, original_configs[2]], 
            width, label='Original', color='lightcoral', alpha=0.7)
    ax2.bar([i + width/2 for i in x], [improved_configs[0], 224, improved_configs[2]], 
            width, label='Improved', color='lightgreen', alpha=0.7)
    
    ax2.set_title('🔧 Configuration Changes', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.legend()
    
    # 3. Metrics Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    original_metrics = [
        results['original']['final_metrics']['accuracy'] * 100,
        results['original']['final_metrics']['precision'] * 100,
        results['original']['final_metrics']['recall'] * 100,
        results['original']['final_metrics']['f1'] * 100
    ]
    improved_metrics = [
        results['improved']['final_metrics']['accuracy'] * 100,
        results['improved']['final_metrics']['precision'] * 100,
        results['improved']['final_metrics']['recall'] * 100,
        results['improved']['final_metrics']['f1'] * 100
    ]
    
    x = range(len(metrics))
    ax3.bar([i - width/2 for i in x], original_metrics, width, label='Original', color='lightcoral', alpha=0.7)
    ax3.bar([i + width/2 for i in x], improved_metrics, width, label='Improved', color='lightgreen', alpha=0.7)
    
    ax3.set_title('📊 All Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.set_ylim(90, 100)
    
    # 4. Improvement Summary
    improvements = {
        'Accuracy': comparison_stats['accuracy_improvement'],
        'F1-Score': comparison_stats['f1_improvement'] * 100,
    }
    
    colors = ['green' if v > 0 else 'red' for v in improvements.values()]
    bars4 = ax4.bar(improvements.keys(), improvements.values(), color=colors, alpha=0.7)
    ax4.set_title('📈 Improvement Summary', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Improvement (%)')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars4, improvements.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if bar.get_height() > 0 else bar.get_height() - 0.05, 
                f'{imp:+.2f}%', ha='center', va='bottom' if bar.get_height() > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Comparison plot saved: results/comparison_analysis.png")

def main():
    print("="*60)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("="*60)
    
    # Load results
    results = load_results()
    if not results:
        return
    
    # Compare metrics
    comparison_stats = compare_metrics(results)
    
    # Create visualization
    create_comparison_plot(results, comparison_stats)
    
    # Summary and recommendations
    print(f"\n" + "="*60)
    print(f"📋 SUMMARY & RECOMMENDATIONS")
    print(f"="*60)
    
    if comparison_stats['accuracy_improvement'] > 0:
        print(f"✅ ACCURACY IMPROVED by {comparison_stats['accuracy_improvement']:.2f}%")
    else:
        print(f"❌ ACCURACY DECREASED by {abs(comparison_stats['accuracy_improvement']):.2f}%")
    
    if comparison_stats['improved_accuracy'] > 95:
        print(f"🎉 EXCELLENT performance achieved!")
    elif comparison_stats['improved_accuracy'] > 90:
        print(f"🎊 GREAT performance!")
    else:
        print(f"🔧 Consider further improvements")
    
    print(f"\n🔍 ANALYSIS:")
    print(f"  • Image size change (200→224): Better EfficientNet compatibility")
    print(f"  • More epochs: Allows model to learn better")
    print(f"  • Patient early stopping: Prevents premature termination")
    print(f"  • Learning rate scheduler: Adaptive learning")
    
    if comparison_stats['accuracy_improvement'] > 0:
        print(f"\n🚀 NEXT STEPS:")
        print(f"  • Monitor for overfitting in longer training")
        print(f"  • Consider data augmentation improvements")
        print(f"  • Try different model architectures")
        print(f"  • Implement cross-validation")
    
    print(f"\n" + "="*60)

if __name__ == "__main__":
    main() 