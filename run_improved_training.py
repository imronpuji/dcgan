#!/usr/bin/env python3
"""
🚀 IMPROVED TRAINING SCRIPT 
=============================
Script ini menjalankan training dengan perbaikan untuk meningkatkan akurasi:
- NUM_EPOCHS ditingkatkan ke 25
- Image size dikembalikan ke 224x224 
- Early stopping lebih patient (15)
- Learning rate scheduler
- Better monitoring
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime
import json

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from dataset import prepare_data_loaders
from models.efficientnet import setup_efficientnet
from training import train_model
from evaluation import (
    evaluate_model, 
    create_evaluation_visualizations,
    plot_training_history
)

def validate_dataset():
    """Validasi dataset sebelum training"""
    print("🔍 VALIDATING DATASET...")
    
    # Check original dataset
    original_path = BASE_DIR
    if not os.path.exists(original_path):
        print(f"❌ Original dataset not found: {original_path}")
        return False
    
    # Count images per class
    classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(original_path, class_name)
        if os.path.exists(class_path):
            images = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  📁 {class_name}: {images} images")
            total_images += images
        else:
            print(f"  ❌ {class_name}: folder not found")
    
    print(f"  📊 Total original images: {total_images}")
    
    # Check augmented dataset if exists
    augmented_path = "augmented_corn_leaf"
    if os.path.exists(augmented_path):
        augmented_total = 0
        for class_name in classes:
            class_path = os.path.join(augmented_path, class_name)
            if os.path.exists(class_path):
                images = len([f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                augmented_total += images
        print(f"  📊 Total augmented images: {augmented_total}")
        print(f"  📊 Total combined images: {total_images + augmented_total}")
    else:
        print(f"  ⚠️  No augmented dataset found")
    
    return total_images > 0

def main():
    print("="*60)
    print("🚀 IMPROVED CORN DISEASE CLASSIFICATION TRAINING")
    print("="*60)
    
    # Validate environment
    print(f"\n📋 ENVIRONMENT INFO:")
    print(f"  🐍 Python: {sys.version.split()[0]}")
    print(f"  🔥 PyTorch: {torch.__version__}")
    print(f"  🖥️  Device: CPU (forced)")
    print(f"  🧠 CPU Cores: {os.cpu_count()}")
    
    # Validate dataset
    if not validate_dataset():
        print("❌ Dataset validation failed!")
        return
    
    # Force CPU usage and utilize all available cores
    device = torch.device("cpu")
    num_cores = os.cpu_count()
    torch.set_num_threads(num_cores)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"\n🔧 TRAINING CONFIGURATION:")
    print(f"  📊 Epochs: {NUM_EPOCHS}")
    print(f"  📦 Batch Size: {BATCH_SIZE}")
    print(f"  📈 Learning Rate: {LEARNING_RATE}")
    print(f"  ⏱️  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  🖼️  Image Size: 224x224")
    print(f"  🏷️  Classes: {NUM_CLASSES}")
    
    # Create directories
    create_directories()
    
    # Define class names
    class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    
    # Prepare data
    print(f"\n🔄 LOADING DATA...")
    try:
        train_loader, val_loader, test_loader = prepare_data_loaders(BASE_DIR)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Setup model
    print(f"\n🧠 SETTING UP MODEL...")
    try:
        model = setup_efficientnet(NUM_CLASSES).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  📊 Total parameters: {total_params:,}")
        print(f"  🔧 Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Error setting up model: {e}")
        return
    
    # Train model
    print(f"\n🏋️ STARTING TRAINING...")
    start_time = datetime.now()
    
    try:
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            device=device,
            save_dir=f'{RESULTS_DIR}/model_training'
        )
        
        training_time = datetime.now() - start_time
        print(f"\n⏱️ Training completed in {training_time}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Plot training history
    print(f"\n📊 CREATING TRAINING PLOTS...")
    try:
        plot_training_history(history, f'{RESULTS_DIR}/model_training')
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")
    
    # Evaluate model
    print(f"\n📈 EVALUATING MODEL...")
    try:
        pred_labels, true_labels = evaluate_model(model, test_loader, device)
        unique_classes = np.unique(true_labels)
        actual_class_names = [class_names[i] for i in unique_classes]
        
        results, metrics = create_evaluation_visualizations(
            true_labels=true_labels,
            pred_labels=pred_labels,
            class_names=actual_class_names,
            save_dir=f'{RESULTS_DIR}/model_evaluation'
        )
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return
    
    # Save training history
    try:
        with open(f'{RESULTS_DIR}/model_training/history.json', 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error saving history: {e}")
    
    # Print final results
    print(f"\n" + "="*60)
    print(f"🎯 FINAL RESULTS:")
    print(f"="*60)
    print(f"📊 Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"📊 Test Precision: {metrics['precision']:.4f}")
    print(f"📊 Test Recall: {metrics['recall']:.4f}")
    print(f"📊 Test F1 Score: {metrics['f1']:.4f}")
    print(f"⏱️ Training Time: {training_time}")
    print(f"📈 Epochs Completed: {len(history['train_loss'])}")
    
    # Analyze improvement
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
    
    print(f"\n📈 TRAINING PROGRESSION:")
    print(f"  🏋️ Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"  ✅ Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"  🎯 Test Accuracy: {metrics['accuracy']*100:.2f}%")
    
    if metrics['accuracy'] > 0.95:
        print(f"\n🎉 EXCELLENT! Accuracy > 95%")
    elif metrics['accuracy'] > 0.90:
        print(f"\n🎊 GREAT! Accuracy > 90%")
    elif metrics['accuracy'] > 0.85:
        print(f"\n👍 GOOD! Accuracy > 85%")
    else:
        print(f"\n🔧 NEEDS IMPROVEMENT - Consider more epochs or different approach")
    
    # Save execution information
    execution_info = {
        'improved_training': True,
        'training_time': str(training_time),
        'device': str(device),
        'cpu_cores_used': num_cores,
        'pytorch_threads': torch.get_num_threads(),
        'num_epochs_configured': NUM_EPOCHS,
        'num_epochs_completed': len(history['train_loss']),
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'image_size': '224x224',
        'final_metrics': metrics,
        'training_progression': {
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'test_acc': metrics['accuracy'] * 100
        }
    }
    
    try:
        with open(f'{RESULTS_DIR}/improved_execution_info.json', 'w') as f:
            json.dump(execution_info, f, indent=4)
        print(f"\n💾 Results saved to: {RESULTS_DIR}/improved_execution_info.json")
    except Exception as e:
        print(f"⚠️ Error saving execution info: {e}")
    
    print(f"\n" + "="*60)
    print(f"✅ IMPROVED TRAINING COMPLETED SUCCESSFULLY!")
    print(f"="*60)

if __name__ == "__main__":
    main() 