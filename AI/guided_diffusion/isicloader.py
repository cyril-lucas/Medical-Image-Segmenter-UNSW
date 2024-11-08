import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(self, args, data_path, transform=None):

        # Load mapping.csv to get image and ground truth paths
        mapping_file = os.path.join(data_path, 'mapping.csv')
        df = pd.read_csv(mapping_file)
        
        # Extract the image and ground truth file paths
        self.image_paths = df['test_images'].apply(lambda x: os.path.join(data_path, x)).tolist()
        self.mask_paths = df['ground_truth'].apply(lambda x: os.path.join(data_path, x)).tolist()
        
        # Store transform and data path
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        img_path = self.image_paths[index]
        msk_path = self.mask_paths[index]

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # Apply transformations, if specified
        if self.transform:
            # Apply the same random transformations to both the image and mask
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        # Extract image name for reference
        name = os.path.basename(img_path)

        return (img, mask, name)
