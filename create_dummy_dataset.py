import os
from PIL import Image
import numpy as np

DATA_DIR = 'data/raw'
CLASSES = ['chat', 'chien', 'voiture']
NB_PER_CLASS = 20
IMAGE_SIZE = (32, 32)

os.makedirs(DATA_DIR, exist_ok=True)

for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)
    for i in range(NB_PER_CLASS):
        # Génère une image simple : pattern basé sur la classe
        if cls == 'chat':
            arr = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
            arr[..., 0] = np.random.randint(50, 200)  # rouge variable
            arr[..., 1] = np.random.randint(40, 160)
            arr[..., 2] = np.random.randint(30, 120)
        elif cls == 'chien':
            arr = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
            arr[..., 0] = np.random.randint(10, 120)
            arr[..., 1] = np.random.randint(80, 220)
            arr[..., 2] = np.random.randint(60, 200)
        else:
            arr = np.random.randint(0, 255, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)

        img = Image.fromarray(arr)
        img.save(os.path.join(cls_dir, f"{cls}_{i:03d}.jpg"))

print("Dataset dummy cree :", DATA_DIR)
