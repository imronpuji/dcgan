import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_gan_losses_multi_panel(losses_dict, save_path):
    """
    Create a multi-panel plot showing Generator vs Discriminator losses for each class.
    
    Args:
        losses_dict (dict): Dictionary containing losses for each class
            Format: {
                'class_name': {
                    'g_losses': [...],
                    'd_losses': [...]
                }
            }
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 12))
    plt.style.use('seaborn')
    
    # Define class positions in the subplot grid
    class_positions = {
        'Common_Rust': 1,
        'Gray_Leaf_Spot': 2,
        'Blight': 3,
        'Healthy': 4
    }
    
    for class_name, losses in losses_dict.items():
        plt.subplot(2, 2, class_positions[class_name])
        
        # Plot Generator and Discriminator losses
        epochs = range(len(losses['g_losses']))
        plt.plot(epochs, losses['g_losses'], 'b-', label='Generator Loss', linewidth=2)
        plt.plot(epochs, losses['d_losses'], 'r--', label='Discriminator Loss', linewidth=2)
        
        plt.title(f'{class_name.replace("_", " ")}', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gan_losses_multi_panel.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_gan_losses_comparative(losses_dict, save_path):
    """
    Create a single comparative plot showing all classes' losses.
    
    Args:
        losses_dict (dict): Dictionary containing losses for each class
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 8))
    plt.style.use('seaborn')
    
    # Define colors for each class
    colors = {
        'Common_Rust': 'red',
        'Gray_Leaf_Spot': 'blue',
        'Blight': 'green',
        'Healthy': 'orange'
    }
    
    for class_name, losses in losses_dict.items():
        epochs = range(len(losses['g_losses']))
        color = colors[class_name]
        
        # Plot Generator loss (solid line)
        plt.plot(epochs, losses['g_losses'], 
                color=color, linestyle='-', 
                label=f'{class_name.replace("_", " ")} Gen',
                linewidth=2)
        
        # Plot Discriminator loss (dashed line)
        plt.plot(epochs, losses['d_losses'], 
                color=color, linestyle='--', 
                label=f'{class_name.replace("_", " ")} Disc',
                linewidth=2)
    
    plt.title('Comparative GAN Training Losses', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gan_losses_comparative.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(losses_dict, data_sizes, save_path):
    """
    Create and save a summary table of GAN training results.
    
    Args:
        losses_dict (dict): Dictionary containing losses for each class
        data_sizes (dict): Dictionary containing original data sizes
        save_path (str): Path to save the table
    """
    summary_data = []
    
    for class_name, losses in losses_dict.items():
        # Calculate convergence epoch (when loss stabilizes)
        g_losses = np.array(losses['g_losses'])
        d_losses = np.array(losses['d_losses'])
        
        # Define convergence as when the change in loss becomes small
        conv_threshold = 0.01
        conv_window = 10
        
        for i in range(len(g_losses) - conv_window):
            g_std = np.std(g_losses[i:i+conv_window])
            d_std = np.std(d_losses[i:i+conv_window])
            if g_std < conv_threshold and d_std < conv_threshold:
                conv_epoch = i
                break
        else:
            conv_epoch = len(g_losses)
        
        summary_data.append({
            'Class': class_name.replace('_', ' '),
            'Original Data Size': data_sizes[class_name],
            'Final Gen Loss': losses['g_losses'][-1],
            'Final Disc Loss': losses['d_losses'][-1],
            'Convergence Epoch': conv_epoch
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(save_path, 'gan_training_summary.csv'), index=False)
    
    # Create a formatted markdown table
    markdown_table = "| Class | Original Data Size | Final Gen Loss | Final Disc Loss | Convergence Epoch |\n"
    markdown_table += "|-------|-------------------|----------------|-----------------|------------------|\n"
    
    for row in summary_data:
        markdown_table += f"| {row['Class']} | {row['Original Data Size']} | {row['Final Gen Loss']:.3f} | {row['Final Disc Loss']:.3f} | ~{row['Convergence Epoch']} |\n"
    
    with open(os.path.join(save_path, 'gan_training_summary.md'), 'w') as f:
        f.write(markdown_table) 