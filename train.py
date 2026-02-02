import torch
from torch.utils.data import DataLoader
from dataset import LeafDataset
from models.generator import Generator
from models.discriminator import Discriminator
from losses import generator_loss, discriminator_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

dataset = LeafDataset("data/images", "data/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

os.makedirs("outputs", exist_ok=True)

for epoch in range(20):
    for i, (img, mask) in enumerate(loader):
        img, mask = img.to(device), mask.to(device)

        # -------- Generator --------
        fake_mask = G(img)
        fake_mask_resized = F.interpolate(
            fake_mask, size=mask.shape[2:], mode="bilinear", align_corners=False
        )

        g_loss = generator_loss(fake_mask_resized, mask)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # -------- Discriminator --------
        real_pred = D(img, mask)
        fake_pred = D(img, fake_mask_resized.detach())

        d_loss = (
            discriminator_loss(real_pred, torch.ones_like(real_pred)) +
            discriminator_loss(fake_pred, torch.zeros_like(fake_pred))
        )

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # -------- Visualization (only first batch) --------
        if i == 0:
            with torch.no_grad():
                for j in range(min(10, img.size(0))):

                    sample_mask = fake_mask_resized[j, 0].cpu().numpy()
                    sample_real = mask[j, 0].cpu().numpy()
                    sample_img = img[j].permute(1, 2, 0).cpu().numpy()

                    # Normalize image safely
                    sample_img = (sample_img - sample_img.min()) / (
                        sample_img.max() - sample_img.min() + 1e-8
                    )

                    # -------- Heatmap --------
                    heatmap = cv2.normalize(sample_mask, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap = heatmap.astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

                    # -------- Overlay --------
                    overlay = 0.6 * sample_img + 0.4 * heatmap
                    overlay = np.clip(overlay, 0, 1)

                    # -------- Plot --------
                    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

                    axs[0].imshow(sample_img)
                    axs[0].set_title("Original Image")
                    axs[0].axis("off")

                    axs[1].imshow(sample_real, cmap="gray")
                    axs[1].set_title("Ground Truth Mask")
                    axs[1].axis("off")

                    axs[2].imshow(heatmap)
                    axs[2].set_title("Generated Heatmap")
                    axs[2].axis("off")

                    axs[3].imshow(overlay)
                    axs[3].set_title("Overlay")
                    axs[3].axis("off")

                    plt.savefig(f"outputs/epoch_{epoch+1}_sample_{j+1}.png")
                    plt.close(fig)

    print(f"Epoch {epoch+1}: G={g_loss.item():.4f}, D={d_loss.item():.4f}")
