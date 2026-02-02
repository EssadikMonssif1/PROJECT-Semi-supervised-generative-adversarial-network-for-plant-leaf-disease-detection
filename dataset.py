import os
import cv2
import torch
from torch.utils.data import Dataset

class LeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, size=128):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size

        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # ---- Load image ----
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.resize(image, (self.size, self.size))
        image = image / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()

        # ---- Load mask ----
        if self.mask_dir is not None:
            base_name = os.path.splitext(img_name)[0]

            # try common mask extensions
            for ext in [".png", ".jpg", ".jpeg"]:
                mask_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(mask_path):
                    break
            else:
                raise FileNotFoundError(f"Mask not found for image: {img_name}")

            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (self.size, self.size))
            mask = mask / 255.0
            mask = torch.tensor(mask).unsqueeze(0).float()

            return image, mask

        return image
