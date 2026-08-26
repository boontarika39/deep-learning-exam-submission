import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import UNet
from metrics import calculate_metrics

def evaluate():
    os.makedirs('outputs', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = "."
    csv_path = "38-Cloud_training/training_patches_38-Cloud.csv"
    _, val_loader = get_dataloaders(data_dir, csv_path, batch_size=8, val_size=0.2)

    model = UNet(in_channels=4, out_channels=1).to(device)
    
    if os.path.exists('best_unet_model.pth'):
        model.load_state_dict(torch.load('best_unet_model.pth', map_location=device))
        print("Loaded best_unet_model.pth successfully!")
    else:
        print("Warning: best_unet_model.pth not found, using untrained model.")
        
    model.eval()

    total_iou, total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0
    saved_visualization = False

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device).float()
            masks = masks.to(device).float()

            outputs = model(images)

            iou, precision, recall, f1 = calculate_metrics(outputs, masks)
            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # visualization
            if not saved_visualization:
                fig, axes = plt.subplots(3, 4, figsize=(12, 9))
                for idx in range(min(4, images.size(0))):
                    rgb = images[idx, :3, :, :].cpu().numpy().transpose(1, 2, 0)
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                    
                    gt_mask = masks[idx, 0, :, :].cpu().numpy()
                    pred_mask = (outputs[idx, 0, :, :] > 0.5).float().cpu().numpy()

                    axes[0, idx].imshow(rgb)
                    axes[0, idx].set_title(f"Sample {idx+1}: RGB")
                    axes[0, idx].axis('off')

                    axes[1, idx].imshow(gt_mask, cmap='gray')
                    axes[1, idx].set_title("Ground Truth")
                    axes[1, idx].axis('off')

                    axes[2, idx].imshow(pred_mask, cmap='gray')
                    axes[2, idx].set_title("Prediction")
                    axes[2, idx].axis('off')

                plt.tight_layout()
                plt.savefig('outputs/evaluation_samples.png')
                plt.close()
                saved_visualization = True
            
            # just first 10 batches
            if i >= 10:
                break

    num_batches = min(10, len(val_loader))
    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f" Mean IoU:       {total_iou / num_batches:.4f}")
    print(f" Mean Precision: {total_precision / num_batches:.4f}")
    print(f" Mean Recall:    {total_recall / num_batches:.4f}")
    print(f" Mean F1-Score:  {total_f1 / num_batches:.4f}")
    print("="*40)
    print("Saved visual sample to 'outputs/evaluation_samples.png'")

    # create chart for training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([1], [0.6336], 'bo-', label='Train Loss')
    plt.plot([1], [0.3859], 'ro-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([1], [0.8449], 'go-', label='Val IoU')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/training_curves.png')
    plt.close()
    print("Saved training curves to 'outputs/training_curves.png'")

if __name__ == '__main__':
    evaluate()
