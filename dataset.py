import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class ColonoscopyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        # Get sorted lists of files to ensure images and masks match up correctly
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load image and mask paths
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # 2. Read images using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Masks are 1-channel grayscale

        # 3. Resize frames to a standard size for the U-Net (256x256 is standard here)
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)

        # 4. Normalize pixel values (Scale 0-255 to 0.0-1.0)
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Binarize the mask to ensure strictly 0 or 1 values (removes blur from resizing)
        mask = (mask > 0.5).astype(np.float32)

        # 5. Format for PyTorch (Expects Channels-First format: Channels x Height x Width)
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0) # Add channel dimension -> (1, H, W)

        # 6. Convert to PyTorch Tensors
        return torch.tensor(image), torch.tensor(mask)

# --- Quick Test to Verify Preprocessing ---
if __name__ == "__main__":
    # Ensure these paths point to where you saved the data
    IMAGE_DIR = "data/images"
    MASK_DIR = "data/masks"
    
    # Initialize dataset
    dataset = ColonoscopyDataset(IMAGE_DIR, MASK_DIR)
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Get a single sample
    img, mask = dataset[0]
    print(f"Image tensor shape: {img.shape}") # Expected: [3, 256, 256]
    print(f"Mask tensor shape: {mask.shape}") # Expected: [1, 256, 256]
    
    # Plotting to verify they match spatially 
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convert tensor back to numpy for matplotlib (Height, Width, Channels)
    img_np = np.transpose(img.numpy(), (1, 2, 0))
    mask_np = mask.squeeze().numpy()
    
    ax[0].imshow(img_np)
    ax[0].set_title("Original Colonoscopy Frame")
    ax[0].axis('off')
    
    ax[1].imshow(mask_np, cmap='gray')
    ax[1].set_title("Polyp Mask (Ground Truth)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()