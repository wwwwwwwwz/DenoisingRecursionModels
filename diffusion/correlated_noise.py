import torch
import torch.nn.functional as F
from torch.distributions import Normal

def get_gaussian_kernel(kernel_size=5, sigma=2.0, device='cpu'):
    # Create a 2D Gaussian kernel normalized to sum to 1
    center = kernel_size // 2
    x, y = torch.meshgrid(
        torch.arange(kernel_size, device=device),
        torch.arange(kernel_size, device=device)
    )
    kernel = torch.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, k, k)

def get_correlated_noise(labels, device):
# Assuming batch["labels"].shape is (batch_size, 900, 1)
# and device is defined

    batch_size = labels.shape[0]
    height, width = 30, 30  # Grid dimensions
    assert height * width == labels.shape[1], "Grid dimensions must match token count"

    # Generate iid standard normal noise in 2D grid shape
    iid_noise = torch.randn(batch_size, 1, height, width, device=device)

    # Define Gaussian kernel for blurring (adjust kernel_size and sigma for desired correlation strength)
    kernel_size = 5
    sigma = 1.5  # Larger sigma means stronger spatial correlation
    kernel = get_gaussian_kernel(kernel_size, sigma, device=device)

    # Apply Gaussian blur with padding to preserve shape
    padding = kernel_size // 2
    blurred = F.conv2d(iid_noise, kernel, padding=padding)

    # Standardize per example (mean ~0, std to 1)
    mean = blurred.mean(dim=[2, 3], keepdim=True)
    std = blurred.std(dim=[2, 3], keepdim=True)
    standardized = (blurred - mean) / (std + 1e-8)  # Avoid division by zero, though unlikely

    # Apply Normal CDF to get correlated uniform [0,1] noise
    normal_dist = Normal(0, 1)
    uniform_noise = normal_dist.cdf(standardized)

    # Reshape back to (batch_size, 900, 1)
    noise = uniform_noise.squeeze(1).view(batch_size, 900, 1)
    return noise

# if __name__ == '__main__':
#     # --- Example Usage ---
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Your batch["labels"].shape is (B, N, 1)
#     # You must determine H and W such that H * W = N
    
#     BATCH_SIZE = 4
#     GRID_HEIGHT = 32  # Example: 32x32 grid
#     GRID_WIDTH = 32   # Example: 32x32 grid
#     NUM_TOKENS = GRID_HEIGHT * GRID_WIDTH
    
#     print(f"Generating noise for shape: ({BATCH_SIZE}, {NUM_TOKENS}, 1)")
#     print(f"Interpreted as a 2D grid of: {GRID_HEIGHT}x{GRID_WIDTH}")
#     print(f"Using device: {device}")

#     # Generate correlated uniform noise with coarse features
#     coarse_noise = generate_correlated_uniform_noise(
#         BATCH_SIZE, GRID_HEIGHT, GRID_WIDTH, device, coarseness=8
#     )

#     # Generate correlated uniform noise with finer features
#     fine_noise = generate_correlated_uniform_noise(
#         BATCH_SIZE, GRID_HEIGHT, GRID_WIDTH, device, coarseness=2
#     )
    
#     # Generate standard uniform noise for comparison
#     standard_noise = torch.rand(BATCH_SIZE, NUM_TOKENS, 1, device=device)

#     print(f"\nShape of coarse noise: {coarse_noise.shape}")
#     print(f"Shape of fine noise: {fine_noise.shape}")
#     print(f"Shape of standard noise: {standard_noise.shape}")

#     # --- Verification ---
    
#     # 1. Verify Distribution (should be uniform)
#     # We check the mean, min, and max. For a uniform[0,1] distribution,
#     # the mean should be ~0.5, min ~0.0, max ~1.0.
#     print(f"\n--- Distribution Stats (should be near 0.5, 0.0, 1.0) ---")
#     print(f"Coarse Noise: Mean={coarse_noise.mean():.4f}, Min={coarse_noise.min():.4f}, Max={coarse_noise.max():.4f}")
#     print(f"Fine Noise:   Mean={fine_noise.mean():.4f}, Min={fine_noise.min():.4f}, Max={fine_noise.max():.4f}")
#     print(f"Std. Noise:   Mean={standard_noise.mean():.4f}, Min={standard_noise.min():.4f}, Max={standard_noise.max():.4f}")

#     # 2. Verify Correlation (optional, requires matplotlib)
#     try:
#         import matplotlib.pyplot as plt
        
#         print("\nPlotting noise for visualization (if matplotlib is installed)...")
        
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#         fig.suptitle("Noise Visualization (First 2 Batch Items)", fontsize=16)

#         for i in range(2):
#             ax_coarse = axes[i, 0]
#             ax_fine = axes[i, 1]
#             ax_std = axes[i, 2]

#             coarse_img = coarse_noise[i].view(GRID_HEIGHT, GRID_WIDTH).cpu().numpy()
#             fine_img = fine_noise[i].view(GRID_HEIGHT, GRID_WIDTH).cpu().numpy()
#             std_img = standard_noise[i].view(GRID_HEIGHT, GRID_WIDTH).cpu().numpy()
            
#             ax_coarse.imshow(coarse_img, cmap='viridis', vmin=0, vmax=1)
#             ax_coarse.set_title(f"Batch {i}: Coarse Correlated Noise")
#             ax_coarse.set_xticks([])
#             ax_coarse.set_yticks([])

#             ax_fine.imshow(fine_img, cmap='viridis', vmin=0, vmax=1)
#             ax_fine.set_title(f"Batch {i}: Fine Correlated Noise")
#             ax_fine.set_xticks([])
#             ax_fine.set_yticks([])

#             ax_std.imshow(std_img, cmap='viridis', vmin=0, vmax=1)
#             ax_std.set_title(f"Batch {i}: Standard Uniform Noise")
#             ax_std.set_xticks([])
#             ax_std.set_yticks([])
        
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()

#     except ImportError:
#         print("\nMatplotlib not found. Skipping visualization.")
#         print("To visualize the noise, run: pip install matplotlib")
