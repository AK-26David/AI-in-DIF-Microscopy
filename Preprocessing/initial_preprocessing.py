import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from pathlib import Path

def fishnet_ridge(image_path):
    img = cv2.imread(image_path)
    g = img[:, :, 1]
    g_clean = cv2.GaussianBlur(g, (3, 3), 0)
    fishnet_map = frangi(g_clean, sigmas=range(1, 4), black_ridges=False)
    fishnet_map = (fishnet_map - np.min(fishnet_map)) / (np.max(fishnet_map) - np.min(fishnet_map))
    fishnet_mask = (fishnet_map * 255).astype(np.uint8)
    final_composite = np.zeros_like(img)
    final_composite[:, :, 1] = cv2.normalize(fishnet_mask, None, 0, 255, cv2.NORM_MINMAX)
    background_faded = (g * 0.2).astype(np.uint8)
    final_composite[:, :, 1] = cv2.add(final_composite[:, :, 1], background_faded)
    return img, final_composite, fishnet_mask

def plot(original, composite, mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='magma')
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(composite[:,:,1], cmap='gray')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

folder_path = Path(r'D:\AI IN DIF\PV')

for item in folder_path.iterdir():
    image_path = str(item.resolve())
    og, final, mask = fishnet_ridge(image_path)
    plot(og, final, mask)