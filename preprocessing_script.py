import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImagePreprocessor:
    def __init__(self):
        self.target_size = (224, 224)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
    
    def apply_clahe(self, image):
        img_array = np.array(image)
        enhanced_channels = []
        for i in range(3):
            enhanced_channel = self.clahe.apply(img_array[:, :, i])
            enhanced_channels.append(enhanced_channel)
        enhanced_array = np.stack(enhanced_channels, axis=2)
        return Image.fromarray(enhanced_array.astype(np.uint8))
    
    def preprocess_image(self, image_path):
        # Step 1: Load and resize
        original = Image.open(image_path).convert('RGB')
        resized = original.resize(self.target_size, Image.BILINEAR)
        
        # Step 2: Apply CLAHE
        enhanced = self.apply_clahe(resized)
        
        # Step 3: Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
        
        tensor = transform(enhanced)
        
        return {
            'original': original,
            'resized': resized,
            'enhanced': enhanced,
            'tensor': tensor
        }

def show_before_after(image_path, save_path=None):
    preprocessor = ImagePreprocessor()
    results = preprocessor.preprocess_image(image_path)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Preprocessing: {os.path.basename(image_path)}', fontsize=14)
    
    # Original
    axes[0].imshow(results['original'])
    axes[0].set_title(f'Original\n{results["original"].size}')
    axes[0].axis('off')
    
    # Resized
    axes[1].imshow(results['resized'])
    axes[1].set_title('Resized\n224x224')
    axes[1].axis('off')
    
    # CLAHE Enhanced
    axes[2].imshow(results['enhanced'])
    axes[2].set_title('CLAHE Enhanced\nClip=2.0')
    axes[2].axis('off')
    
    # Final tensor (denormalized for display)
    tensor_display = results['tensor'].permute(1, 2, 0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    denorm = torch.clamp(tensor_display * std + mean, 0, 1)
    
    axes[3].imshow(denorm)
    axes[3].set_title('Normalized Tensor\n(for EfficientNet)')
    axes[3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def run_demo():
    print("PREPROCESSING DEMO - Before/After Comparison")
    print("="*50)
    
    # Find sample images
    classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    for class_name in classes:
        class_dir = f"data/{class_name}"
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                img_path = os.path.join(class_dir, images[0])
                print(f"\nProcessing {class_name}: {images[0]}")
                show_before_after(img_path, f"results/preprocessing_{class_name}.png")
                break

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_demo()
