import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
from config import *
from visualization import plot_gan_losses_multi_panel, plot_gan_losses_comparative, create_summary_table

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, feature_dim=GEN_FEATURES):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Initial block
            nn.ConvTranspose2d(latent_dim, feature_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True),
            
            # Upsampling blocks
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            
            # Output block
            nn.ConvTranspose2d(feature_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, feature_dim=DISC_FEATURES):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Initial block
            nn.Conv2d(3, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsampling blocks
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output block
            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

class DataPreprocessing:
    def __init__(self, image_size=IMG_SIZE):
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self, data_path, class_name):
        dataset = ImageFolder(
            root=os.path.join(data_path, class_name),
            transform=self.transforms
        )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

def train_dcgan_for_class(class_name, data_loader, device, save_dir, generated_dir):
    # Initialize networks
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    # Setup optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=GAN_LEARNING_RATE, betas=(GAN_BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=GAN_LEARNING_RATE, betas=(GAN_BETA1, 0.999))
    
    criterion = nn.BCELoss()
    
    # Lists to store losses
    g_losses = []
    d_losses = []

    # Create directories for saving training progress
    class_progress_dir = os.path.join(generated_dir, f'{class_name}_training_progress')
    os.makedirs(class_progress_dir, exist_ok=True)
    
    # Fixed noise for tracking progress
    fixed_noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=device)
    
    print(f"\nStarting DCGAN training for {class_name}")
    
    for epoch in range(GAN_EPOCHS):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        num_batches = 0
        
        for i, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            netD.zero_grad()
            label_real = torch.ones(batch_size, device=device)
            label_fake = torch.zeros(batch_size, device=device)
            
            output_real = netD(real_images)
            d_loss_real = criterion(output_real, label_real)
            
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            d_loss_fake = criterion(output_fake, label_fake)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizerD.step()
            
            # Train Generator
            netG.zero_grad()
            output_fake = netD(fake_images)
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            optimizerG.step()
            
            # Record losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            num_batches += 1
            
            if i % 100 == 0:
                print(f'[{epoch}/{GAN_EPOCHS}][{i}/{len(data_loader)}] '
                      f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}')
        
        # Average losses for the epoch
        g_losses.append(g_loss_epoch / num_batches)
        d_losses.append(d_loss_epoch / num_batches)
        
        # Save training progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{GAN_EPOCHS}] - '
                  f'G_Loss: {g_losses[-1]:.4f} D_Loss: {d_losses[-1]:.4f}')
            
            # Save model checkpoints
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'g_optimizer_state_dict': optimizerG.state_dict(),
                'd_optimizer_state_dict': optimizerD.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{class_name}_epoch_{epoch+1}.pth'))
            
            # Generate and save sample images
            with torch.no_grad():
                fake_images = netG(fixed_noise).detach().cpu()
                epoch_dir = os.path.join(class_progress_dir, f'epoch_{epoch+1}')
                os.makedirs(epoch_dir, exist_ok=True)
                
                for j, image in enumerate(fake_images):
                    image_path = os.path.join(epoch_dir, f'sample_{j+1}.png')
                    save_image(image, image_path, normalize=True)
                print(f"Saved training progress images for epoch {epoch+1}")
    
    # Save final models
    torch.save(netG.state_dict(), os.path.join(save_dir, f'generator_{class_name}_final.pth'))
    torch.save(netD.state_dict(), os.path.join(save_dir, f'discriminator_{class_name}_final.pth'))
    
    # Save final sample images
    with torch.no_grad():
        final_dir = os.path.join(class_progress_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)
        
        fake_images = netG(fixed_noise).detach().cpu()
        for i, image in enumerate(fake_images):
            image_path = os.path.join(final_dir, f'sample_{i+1}.png')
            save_image(image, image_path, normalize=True)
        print(f"Saved final generated images for {class_name}")

    return {'g_losses': g_losses, 'd_losses': d_losses}

def main():
    # Create necessary directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    models_dir = os.path.join(RESULTS_DIR, 'models')
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    generated_dir = os.path.join(RESULTS_DIR, 'generated_images')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize preprocessing
    preprocessor = DataPreprocessing()
    
    # Dictionary to store losses for all classes
    all_losses = {}
    data_sizes = {}
    
    # Train DCGAN for each class
    classes = ['Common_Rust', 'Gray_Leaf_Spot', 'Blight', 'Healthy']
    
    for class_name in classes:
        print(f"\nProcessing {class_name}")
        data_loader = preprocessor.load_data(BASE_DIR, class_name)
        data_sizes[class_name] = len(data_loader.dataset)
        
        # Train DCGAN
        losses = train_dcgan_for_class(class_name, data_loader, device, models_dir, generated_dir)
        all_losses[class_name] = losses
    
    # Create visualizations
    plot_gan_losses_multi_panel(all_losses, plots_dir)
    plot_gan_losses_comparative(all_losses, plots_dir)
    create_summary_table(all_losses, data_sizes, plots_dir)
    
    print("\nTraining complete! Check the results directory for visualizations and models.")

if __name__ == '__main__':
    main() 