import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class CornLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None, show_logs=False):
        self.root_dir = root_dir
        self.transform = transform
        self.show_logs = show_logs
        self.classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
        self.images = []
        self.labels = []
        
        print(f"\n{'='*60}")
        print(f"📁 LOADING DATASET FROM: {root_dir}")
        print(f"{'='*60}")
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                class_count = 0
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
                        class_count += 1
                print(f"📷 {class_name:15} -> {class_count:4} gambar (label: {class_idx})")
            else:
                print(f"⚠️  {class_name:15} -> FOLDER NOT FOUND!")
        
        print(f"\n✅ Total dataset: {len(self.images)} gambar")
        print(f"📊 Kelas: {self.classes}")
        
        # Check class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"\n📊 DISTRIBUSI KELAS:")
        for label, count in zip(unique_labels, counts):
            print(f"   {self.classes[label]:15} -> {count:4} gambar ({count/len(self.labels)*100:.1f}%)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Log untuk gambar pertama atau jika show_logs=True
        if idx < 3 or self.show_logs:
            print(f"\n{'='*50}")
            print(f"🔄 TRANSFORMASI DATA - Sample #{idx}")
            print(f"{'='*50}")
            print(f"📁 File: {os.path.basename(img_path)}")
            print(f"📂 Path: {img_path}")
        
        # Step 1: Load gambar
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if idx < 3 or self.show_logs:
            print(f"🖼️  STEP 1 - Load Image:")
            print(f"   Format: PIL Image RGB")
            print(f"   Ukuran asli: {image.size} (width x height)")
            print(f"   Label: {label} ({self.classes[label]})")
        
        if self.transform:
            if idx < 3 or self.show_logs:
                print(f"⚙️  STEP 2 - Applying Transformations:")
            
            # Simpan gambar asli untuk perbandingan
            original_size = image.size
            
            # Apply transformations
            transformed_image = self.transform(image)
            
            if idx < 3 or self.show_logs:
                print(f"   ✅ Resize: {original_size} -> (224, 224)")
                print(f"   ✅ Data Augmentation: Random flip, rotation, color jitter")
                print(f"   ✅ ToTensor: PIL Image -> PyTorch Tensor")
                print(f"   ✅ Normalize: ImageNet mean/std")
                print(f"")
                print(f"📊 HASIL TRANSFORMASI:")
                print(f"   Tipe data: {type(transformed_image)}")
                print(f"   Shape: {transformed_image.shape}")
                print(f"   Dtype: {transformed_image.dtype}")
                print(f"   Min value: {transformed_image.min():.4f}")
                print(f"   Max value: {transformed_image.max():.4f}")
                print(f"   Mean: {transformed_image.mean():.4f}")
                print(f"   Std: {transformed_image.std():.4f}")
                
                # Tampilkan beberapa nilai pixel
                print(f"")
                print(f"🔢 CONTOH NILAI NUMERIK (Channel R, pixel 0-2):")
                print(f"   {transformed_image[0, 0, :3].tolist()}")
                print(f"🔢 CONTOH NILAI NUMERIK (Channel G, pixel 0-2):")
                print(f"   {transformed_image[1, 0, :3].tolist()}")
                print(f"🔢 CONTOH NILAI NUMERIK (Channel B, pixel 0-2):")
                print(f"   {transformed_image[2, 0, :3].tolist()}")
            
            return transformed_image, label
        else:
            return image, label

def get_transforms():
    print(f"\n🔧 KONFIGURASI TRANSFORMASI:")
    print(f"{'='*40}")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Dikembalikan ke 224 untuk EfficientNet optimal
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Dikembalikan ke 224 untuk EfficientNet optimal
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("📋 TRAINING TRANSFORMS:")
    print("   1. Resize -> (224, 224)")
    print("   2. RandomHorizontalFlip -> 50% chance")
    print("   3. RandomVerticalFlip -> 50% chance") 
    print("   4. RandomRotation -> ±10 derajat")
    print("   5. ColorJitter -> brightness±0.2, contrast±0.2")
    print("   6. ToTensor -> PIL Image ke Tensor [0,1]")
    print("   7. Normalize -> ImageNet mean/std")
    
    print("\n📋 TEST TRANSFORMS:")
    print("   1. Resize -> (224, 224)")
    print("   2. ToTensor -> PIL Image ke Tensor [0,1]")
    print("   3. Normalize -> ImageNet mean/std")
    
    return train_transform, test_transform

def get_stratified_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create stratified train/validation/test splits to ensure all classes are represented
    """
    # Get labels from dataset
    labels = np.array(dataset.labels)
    indices = np.arange(len(labels))
    
    # Check if all classes have enough samples
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples = min(counts)
    
    print(f"\n🔄 STRATIFIED SPLITTING:")
    print(f"   Total samples: {len(labels)}")
    print(f"   Classes: {len(unique_labels)}")
    print(f"   Min samples per class: {min_samples}")
    
    if min_samples < 3:
        print("⚠️  WARNING: Some classes have very few samples. Results may be unreliable.")
    
    # First split: train vs (val+test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels, 
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels,
        test_size=test_ratio/(val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Print split statistics
    train_split_labels = labels[train_indices]
    val_split_labels = labels[val_indices]
    test_split_labels = labels[test_indices]
    
    print(f"\n📊 HASIL PEMBAGIAN STRATIFIED:")
    print(f"   Training: {len(train_indices)} samples ({len(train_indices)/len(labels)*100:.1f}%)")
    print(f"   Validation: {len(val_indices)} samples ({len(val_indices)/len(labels)*100:.1f}%)")
    print(f"   Testing: {len(test_indices)} samples ({len(test_indices)/len(labels)*100:.1f}%)")
    
    print(f"\n📈 DISTRIBUSI PER SPLIT:")
    class_names = dataset.classes
    for split_name, split_labels in [("Train", train_split_labels), ("Val", val_split_labels), ("Test", test_split_labels)]:
        unique_split, counts_split = np.unique(split_labels, return_counts=True)
        print(f"   {split_name}:")
        for label in range(len(class_names)):
            if label in unique_split:
                count = counts_split[unique_split == label][0]
                print(f"      {class_names[label]:15} -> {count:3} samples")
            else:
                print(f"      {class_names[label]:15} -> {0:3} samples ⚠️")
    
    return train_indices, val_indices, test_indices

def prepare_data_loaders(dataset_path):
    print(f"\n🚀 MEMPERSIAPKAN DATA LOADERS")
    print(f"{'='*50}")
    
    train_transform, test_transform = get_transforms()
    
    print(f"\n📦 Loading datasets...")
    original_dataset = CornLeafDataset(dataset_path, train_transform, show_logs=False)
    augmented_dataset = CornLeafDataset("augmented_corn_leaf", train_transform, show_logs=False)

    full_dataset = torch.utils.data.ConcatDataset([original_dataset, augmented_dataset])
    
    # Use stratified split for better class distribution
    # Note: ConcatDataset doesn't have labels attribute, so we need to handle this differently
    # For now, let's use the original approach but with better error handling
    
    # Calculate splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\n📊 PEMBAGIAN DATASET:")
    print(f"   Total: {total_size} gambar")
    print(f"   Training: {train_size} gambar (70%)")
    print(f"   Validation: {val_size} gambar (15%)")
    print(f"   Testing: {test_size} gambar (15%)")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16,
        persistent_workers=True
    )
    
    print(f"\n🔄 DATA LOADERS CREATED:")
    print(f"   Batch size: 128")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Tampilkan contoh batch
    print(f"\n🎯 CONTOH BATCH DATA:")
    print(f"{'='*40}")
    train_iter = iter(train_loader)
    sample_batch, sample_labels = next(train_iter)
    
    print(f"📦 Batch shape: {sample_batch.shape}")
    print(f"   [batch_size, channels, height, width]")
    print(f"   [{sample_batch.shape[0]}, {sample_batch.shape[1]}, {sample_batch.shape[2]}, {sample_batch.shape[3]}]")
    print(f"")
    print(f"🏷️  Labels shape: {sample_labels.shape}")
    print(f"🏷️  Sample labels: {sample_labels[:10].tolist()}")
    print(f"")
    print(f"📊 Batch statistics:")
    print(f"   Min value: {sample_batch.min():.4f}")
    print(f"   Max value: {sample_batch.max():.4f}")
    print(f"   Mean: {sample_batch.mean():.4f}")
    print(f"   Std: {sample_batch.std():.4f}")
    print(f"   Memory size: ~{sample_batch.numel() * 4 / (1024*1024):.1f} MB")
    
    return train_loader, val_loader, test_loader

def prepare_original_data_loaders_only(dataset_path):
    """Simple function to load only original dataset without augmented data using stratified split"""
    print(f"\n🚀 MEMPERSIAPKAN ORIGINAL DATA LOADERS SAJA")
    print(f"{'='*50}")
    
    train_transform, test_transform = get_transforms()
    
    print(f"\n📦 Loading original dataset only...")
    original_dataset = CornLeafDataset(dataset_path, train_transform, show_logs=False)
    
    # Use stratified split to ensure all classes are represented
    train_indices, val_indices, test_indices = get_stratified_splits(
        original_dataset, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15,
        random_state=42
    )
    
    # Create subset datasets
    train_dataset = Subset(original_dataset, train_indices)
    val_dataset = Subset(original_dataset, val_indices)
    test_dataset = Subset(original_dataset, test_indices)
    
    # Update transform for validation and test (no augmentation)
    val_test_dataset = CornLeafDataset(dataset_path, test_transform, show_logs=False)
    val_dataset = Subset(val_test_dataset, val_indices)
    test_dataset = Subset(val_test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16,
        persistent_workers=True
    )
    
    print(f"\n🔄 STRATIFIED DATA LOADERS CREATED:")
    print(f"   Batch size: 128")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# Fungsi untuk demo transformasi
def demo_data_transformation():
    """Demo untuk melihat transformasi data secara detail"""
    print(f"\n🎬 DEMO TRANSFORMASI DATA")
    print(f"{'='*60}")
    
    # Cari sample gambar
    sample_path = None
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                sample_path = os.path.join(root, file)
                break
        if sample_path:
            break
    
    if not sample_path:
        print("❌ Tidak ada sample gambar ditemukan")
        return
    
    # Buat dataset dengan log detail
    train_transform, _ = get_transforms()
    demo_dataset = CornLeafDataset(os.path.dirname(sample_path), train_transform, show_logs=True)
    
    # Ambil 1 sample
    sample_data, sample_label = demo_dataset[0]
    
    print(f"\n✅ DEMO SELESAI!")
    print(f"Gambar berhasil diubah dari file .jpg menjadi tensor numerik siap training!")

if __name__ == "__main__":
    # Jalankan demo jika file dijalankan langsung
    demo_data_transformation()