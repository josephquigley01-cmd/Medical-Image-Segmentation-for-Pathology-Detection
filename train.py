import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

# Import custom classes
from dataset import ColonoscopyDataset
from model import UNet

# --- 1. Hyperparameters & Device Setup ---
# Automatically use NVIDIA GPU, Apple Silicon (MPS), or fallback to CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

LEARNING_RATE = 1e-4
BATCH_SIZE = 8       # If get "Out of Memory" error, reduce this to 4 or 2
NUM_EPOCHS = 5       # Set to 5 for a quick test. For final portfolio results, use 20-30+ epochs
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# --- 2. Evaluation Metric (Intersection over Union) ---
def calculate_iou(predictions, targets, smooth=1e-6):
    """
    Calculates the Intersection over Union (IoU) metric.
    """
    # Apply sigmoid to convert raw logits to probabilities (0 to 1)
    preds = torch.sigmoid(predictions)
    # Threshold at 0.5 to make them strictly binary (0 or 1)
    preds = (preds > 0.5).float()
    
    # Flatten the tensors to compute the metric across the whole batch
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection and union
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    
    # Add smoothing to prevent division by zero
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

# --- 3. Visualization Function ---
def visualize_prediction(model, dataset):
    """
    Pulls a random image from the validation set, runs it through the trained model,
    and plots the original image, ground truth mask, and predicted mask.
    """
    model.eval()
    
    # Pick a random sample from the validation set
    idx = np.random.randint(0, len(dataset))
    img_tensor, mask_tensor = dataset[idx]
    
    # Add batch dimension and send to device
    img_input = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        pred_mask = model(img_input)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > 0.5).float().cpu().squeeze().numpy()
        
    # Convert tensors to numpy for plotting
    img_vis = np.transpose(img_tensor.numpy(), (1, 2, 0))
    mask_vis = mask_tensor.squeeze().numpy()
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_vis)
    axes[0].set_title("Original Colonoscopy")
    axes[0].axis("off")
    
    axes[1].imshow(mask_vis, cmap="gray")
    axes[1].set_title("Ground Truth (Actual Polyp)")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("U-Net Prediction")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("portfolio_showcase.png")
    print("\nSaved visualization to 'portfolio_showcase.png'.")
    plt.show()

# --- 4. Main Training Script ---
def main():
    # Load full dataset
    full_dataset = ColonoscopyDataset(IMAGE_DIR, MASK_DIR)
    
    # Split into 80% Training and 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create DataLoaders to batch and shuffle the data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model, Loss Function, and Optimizer
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # BCEWithLogitsLoss is mathematically more stable than combining Sigmoid + BCELoss manually
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training on {len(train_dataset)} images, validating on {len(val_dataset)} images...")
    
    best_iou = 0.0

    # --- 5. Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass
            predictions = model(data)
            loss = criterion(predictions, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad() # Clear old gradients
            loss.backward()       # Calculate new gradients
            optimizer.step()      # Update weights
            
            train_loss += loss.item()
            
        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode (turns off Dropout/BatchNorm updates)
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad(): # Don't track gradients during validation to save memory
            for data, targets in val_loader:
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                
                predictions = model(data)
                loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                val_iou += calculate_iou(predictions, targets)
                
        # Calculate averages for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # Save the best model weights
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "unet_polyp_model.pth")
            print("  -> New best model saved!")

    print(f"Training complete! Best Validation IoU: {best_iou:.4f}")
    
    # --- 6. Portfolio Visualization Execution ---
    # Load the best weights before visualization
    model.load_state_dict(torch.load("unet_polyp_model.pth", weights_only=True))
    visualize_prediction(model, val_dataset)


if __name__ == "__main__":
    main()