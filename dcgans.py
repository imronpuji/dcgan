import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from torchvision.utils import save_image
import time
import os
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

# Data preprocessing
class DataPreprocessing:
    def __init__(self, image_size=128): # Changed image size from 64 to 128
        print("\n[INFO] Initializing Data Preprocessing...")
        self.image_size = image_size
        
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self, data_path, batch_size=32):
        print(f"\n[INFO] Loading dataset from: {data_path}")
        
        if not os.path.exists(data_path):
            raise RuntimeError(f"Dataset path {data_path} does not exist")
        
        # Load the entire dataset (without train/test split)
        dataset = ImageFolder(root=data_path, transform=self.transforms)
        print(f"[INFO] Found {len(dataset)} images")
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Print class distribution
        print("\n[INFO] Dataset Class Distribution:")
        for idx, class_name in enumerate(dataset.classes):
            n_samples = len([x for x, y in dataset.samples if y == idx])
            print(f"Class {idx}: {class_name} - {n_samples} images")
            
        return data_loader

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            # State size: 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # State size: 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # State size: 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # State size: 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # State size: 64 x 64 x 64
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: 3 x 128 x 128
        )

    def forward(self, x):
        return self.main(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input size: 3 x 128 x 128
            nn.utils.spectral_norm(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # State size: 64 x 64 x 64
            nn.utils.spectral_norm(
                nn.Conv2d(64, 128, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: 128 x 32 x 32
            nn.utils.spectral_norm(
                nn.Conv2d(128, 256, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: 256 x 16 x 16
            nn.utils.spectral_norm(
                nn.Conv2d(256, 512, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: 512 x 8 x 8
            nn.utils.spectral_norm(
                nn.Conv2d(512, 1024, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: 1024 x 4 x 4
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).squeeze(1)

def train_dcgan(data_loader, class_name, device, nz=100, num_epochs=200):
    # Create the generator and discriminator with optimized settings
    netG = Generator(latent_dim=nz).to(device)
    netD = Discriminator().to(device)
    
    # Enable automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    # Enable cuDNN benchmarking and deterministic mode
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Initialize MSELoss function
    criterion = nn.MSELoss()

    # Create batch of latent vectors for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup optimizers with adjusted learning rates and betas
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.00005, betas=(0.5, 0.999))
    
    # Create directories for saving
    os.makedirs(f"generated_images/{class_name}", exist_ok=True)
    os.makedirs(f"models/{class_name}", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Lists to store losses
    g_losses = []
    d_losses = []
    
    # Training Loop
    print(f"Starting Training Loop for {class_name}...")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # For tracking epoch losses
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        n_batch = 0
        
        for i, data in enumerate(data_loader, 0):
            batch_size = data[0].size(0)
            
            # Move data to GPU
            real_cpu = data[0].to(device, non_blocking=True)
            
            # Use label smoothing for real labels
            real_labels = torch.full((batch_size,), 0.9, dtype=torch.float, device=device)
            fake_labels = torch.zeros((batch_size,), dtype=torch.float, device=device)
            
            # Train discriminator with mixed precision
            optimizerD.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            
            with torch.cuda.amp.autocast():
                # Add noise to discriminator inputs
                real_cpu_noisy = real_cpu + 0.05 * torch.randn_like(real_cpu)
                real_output = netD(real_cpu_noisy)
                d_real_loss = criterion(real_output, real_labels)
                
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                fake_noisy = fake.detach() + 0.05 * torch.randn_like(fake)
                fake_output = netD(fake_noisy)
                d_fake_loss = criterion(fake_output, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
            
            scaler.scale(d_loss).backward()
            scaler.step(optimizerD)
            
            # Train generator with mixed precision
            optimizerG.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                fake_output = netD(fake)
                g_loss = criterion(fake_output, real_labels)
            
            scaler.scale(g_loss).backward()
            scaler.step(optimizerG)
            
            # Update scaler
            scaler.update()
            
            # Record losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            n_batch += 1
            
            if (i+1) % 100 == 0:
                print(f'{class_name} - Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}')
        
        # Calculate average losses for the epoch
        avg_g_loss = g_loss_epoch / n_batch
        avg_d_loss = d_loss_epoch / n_batch
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Save images and models every 5 epochs
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
                save_image(fake.detach(),
                         f'generated_images/{class_name}/fake_samples_epoch_{epoch+1}.png',
                         normalize=True)
                print(f"Saved generated images for {class_name} epoch {epoch+1}")
                
                # Save model checkpoints
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': netG.state_dict(),
                    'discriminator_state_dict': netD.state_dict(),
                    'g_optimizer_state_dict': optimizerG.state_dict(),
                    'd_optimizer_state_dict': optimizerD.state_dict(),
                    'g_losses': g_losses,
                    'd_losses': d_losses,
                    'scaler_state_dict': scaler.state_dict()  # Save scaler state
                }, f'models/{class_name}/checkpoint_epoch_{epoch+1}.pth')
                
                # Plot and save loss curves
                plot_losses(g_losses, d_losses, class_name, epoch+1)
    
    # Save final models
    torch.save(netG.state_dict(), f'models/{class_name}/generator_final.pth')
    torch.save(netD.state_dict(), f'models/{class_name}/discriminator_final.pth')
    
    # Save final loss curves
    plot_losses(g_losses, d_losses, class_name, num_epochs, is_final=True)
    
    return netG, g_losses, d_losses

def plot_losses(g_losses, d_losses, class_name, epoch, is_final=False):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(g_losses) + 1)
    
    # Generator Loss subplot
    ax1.plot(epochs, g_losses, 'b-', label='Generator Loss', linewidth=2)
    ax1.set_title(f'Generator Loss Progress - {class_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add min, max, mean annotations for Generator
    min_g = min(g_losses)
    max_g = max(g_losses)
    mean_g = np.mean(g_losses)
    ax1.annotate(f'Min: {min_g:.4f}\nMax: {max_g:.4f}\nMean: {mean_g:.4f}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='w', ec='k', alpha=0.8),
                verticalalignment='top')
    
    # Discriminator Loss subplot
    ax2.plot(epochs, d_losses, 'r-', label='Discriminator Loss', linewidth=2)
    ax2.set_title(f'Discriminator Loss Progress - {class_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add min, max, mean annotations for Discriminator
    min_d = min(d_losses)
    max_d = max(d_losses)
    mean_d = np.mean(d_losses)
    ax2.annotate(f'Min: {min_d:.4f}\nMax: {max_d:.4f}\nMean: {mean_d:.4f}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='w', ec='k', alpha=0.8),
                verticalalignment='top')
    
    # Add overall title
    plt.suptitle(f'Training Progress for {class_name} (Epoch {epoch})', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    suffix = 'final' if is_final else f'epoch_{epoch}'
    save_path = f'plots/{class_name}_losses_{suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Loss plot saved to {save_path}")

def plot_comparative_losses(all_losses):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Color map for different classes
    colors = ['b', 'r', 'g', 'purple']
    
    # Plot Generator losses
    ax1.set_title('Generator Losses Comparison', fontsize=12)
    for (class_name, (g_losses, _)), color in zip(all_losses.items(), colors):
        epochs = range(1, len(g_losses) + 1)
        ax1.plot(epochs, g_losses, color=color, label=class_name, linewidth=2, alpha=0.7)
        
        # Add final loss value annotation
        final_loss = g_losses[-1]
        ax1.annotate(f'{final_loss:.4f}', 
                    xy=(len(g_losses), final_loss),
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Generator Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot Discriminator losses
    ax2.set_title('Discriminator Losses Comparison', fontsize=12)
    for (class_name, (_, d_losses)), color in zip(all_losses.items(), colors):
        epochs = range(1, len(d_losses) + 1)
        ax2.plot(epochs, d_losses, color=color, label=class_name, linewidth=2, alpha=0.7)
        
        # Add final loss value annotation
        final_loss = d_losses[-1]
        ax2.annotate(f'{final_loss:.4f}', 
                    xy=(len(d_losses), final_loss),
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Discriminator Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add overall title
    plt.suptitle('Comparative Training Progress Across Classes', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot with high resolution
    save_path = 'plots/comparative_losses.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Comparative loss plot saved to {save_path}")

def plot_training_summary(all_losses):
    """Create a summary plot with key metrics for all classes"""
    classes = list(all_losses.keys())
    metrics = {
        'Final G-Loss': [losses[0][-1] for losses in all_losses.values()],
        'Final D-Loss': [losses[1][-1] for losses in all_losses.values()],
        'Mean G-Loss': [np.mean(losses[0]) for losses in all_losses.values()],
        'Mean D-Loss': [np.mean(losses[1]) for losses in all_losses.values()],
        'Min G-Loss': [min(losses[0]) for losses in all_losses.values()],
        'Min D-Loss': [min(losses[1]) for losses in all_losses.values()]
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Number of metrics and classes
    n_metrics = len(metrics)
    n_classes = len(classes)
    
    # Set up bar positions
    bar_width = 0.15
    index = np.arange(n_classes)
    
    # Plot bars for each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        position = index + i * bar_width
        ax.bar(position, values, bar_width, label=metric_name, alpha=0.7)
        
        # Add value labels on top of bars
        for j, value in enumerate(values):
            ax.text(position[j], value, f'{value:.4f}', 
                   ha='center', va='bottom', rotation=45)
    
    # Customize plot
    ax.set_xlabel('Classes')
    ax.set_ylabel('Loss Values')
    ax.set_title('Training Summary Metrics by Class')
    ax.set_xticks(index + bar_width * (n_metrics/2 - 0.5))
    ax.set_xticklabels(classes)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = 'plots/training_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Training summary plot saved to {save_path}")

def load_class_data(preprocessor, data_path, class_name, batch_size=128):  # Decreased from 512 to 128
    # Load the entire dataset
    dataset = ImageFolder(root=data_path, transform=preprocessor.transforms)
    
    # Get indices for the specific class
    class_idx = dataset.class_to_idx[class_name]
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
    
    # Create a subset for this class
    subset = torch.utils.data.Subset(dataset, indices)
    
    # Create dataloader with optimized settings
    data_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Increased from 2 to 4
        pin_memory=True,  # Enable pinned memory for faster GPU transfer
        persistent_workers=True  # Keep workers alive between iterations
    )
    
    print(f"[INFO] Found {len(subset)} images for class {class_name}")
    return data_loader

def main():
    # Create output directories
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Initialize data preprocessing
    preprocessor = DataPreprocessing()
    
    # Define classes
    classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    
    # Dictionary to store all losses
    all_losses = {}
    
    # Train DCGAN for each class
    for class_name in classes:
        print(f"\n[INFO] Processing class: {class_name}")
        
        # Load data for this class
        data_loader = load_class_data(preprocessor, './data', class_name)
        
        # Train DCGAN
        print(f"[INFO] Training DCGAN for {class_name}...")
        generator, g_losses, d_losses = train_dcgan(data_loader, class_name, device)
        all_losses[class_name] = (g_losses, d_losses)
        
        print(f"[INFO] Training completed for {class_name}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Generate all summary plots
    plot_comparative_losses(all_losses)
    plot_training_summary(all_losses)
    print("\n[INFO] All training completed")

if __name__ == '__main__':
    main()
