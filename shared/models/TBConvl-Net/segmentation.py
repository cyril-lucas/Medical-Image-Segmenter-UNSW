# # Run segmentation script
# python segmentation.py --model_pth '/srv/scratch/z5450230/tbconv/model/1modelWeights_Swin_Trans_Weights_Swin_Trans_Leather.pth' --test_images_dir '/srv/scratch/z5450230/dataset/ISIC2018_Task1-2_Validation_Input' --test_masks_dir '/srv/scratch/z5450230/dataset/ISIC2018_Task1_Validation_GroundTruth' --save_dir_pred '/srv/scratch/z5450230/tbconv/output/segmented predicted images' --save_dir_gt '/srv/scratch/z5450230/tbconv/output/segmented ground truth'


import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import argparse
from SwinUnet import SwinUNet
from SegmentationDataset import SegmentationDataset
from torchvision import transforms

# Set up argument parser
parser = argparse.ArgumentParser(description="Segmentation and Saving Predictions")
parser.add_argument('--model_pth', type=str, required=True, help='Path to the saved model directory')
parser.add_argument('--test_images_dir', type=str, required=True, help='Path to the test images directory')
parser.add_argument('--test_masks_dir', type=str, required=True, help='Path to the test masks directory')
parser.add_argument('--save_dir_pred', type=str, default='./segmented_predicted_images', help='Path to save predictions')
parser.add_argument('--save_dir_gt', type=str, default='./segmented_ground_truth', help='Path to save ground truth')

args = parser.parse_args()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = SwinUNet(input_channels=3, output_channels=1).to(device)

# Load state dict, removing 'module.' if necessary
state_dict = torch.load(args.model_pth)
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

# Data loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
test_dataset = SegmentationDataset(args.test_images_dir, args.test_masks_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Save predictions
os.makedirs(args.save_dir_pred, exist_ok=True)
os.makedirs(args.save_dir_gt, exist_ok=True)

with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        pred_mask = outputs.cpu().squeeze().numpy()
        gt_mask = masks.cpu().squeeze().numpy()

        # Save predicted mask
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img.save(os.path.join(args.save_dir_pred, f"{i+1}.png"))

        # Save ground truth mask
        gt_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
        gt_img.save(os.path.join(args.save_dir_gt, f"{i+1}.tiff"))

print("Segmentation Complete!")