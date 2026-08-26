import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import UNet
from metrics import BCEDiceLoss, calculate_metrics

def main():
    os.makedirs('outputs', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    # initial params
    batch_size = 8
    epochs = 10  
    lr = 1e-4

    # load data
    data_dir = "."
    csv_path = "38-Cloud_training/training_patches_38-Cloud.csv"
    train_loader, val_loader = get_dataloaders(data_dir, csv_path, batch_size=batch_size, val_size=0.2)

    # set model, criterion, optimizer
    model = UNet(in_channels=4, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    best_val_iou = 0.0

    print("start the loop...")
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device).float(), masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # validate
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device).float(), masks.to(device).float()
                outputs = model(images)
                
                val_loss += criterion(outputs, masks).item()
                iou, _, _, _ = calculate_metrics(outputs, masks)
                val_iou += iou
                
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # save model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"Val IoU: {best_val_iou:.4f}")

    # plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.title('Validation IoU Curve')

    plt.tight_layout()
    plt.savefig('outputs/training_curves.png') 
    plt.close()

if __name__ == '__main__':
    main()
