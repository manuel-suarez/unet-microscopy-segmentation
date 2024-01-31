"""
Mitochondria semantic segmentation using U-net, Attention Unet and Att Res Unet
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

home_dir = os.path.expanduser('~')
base_dir = os.path.join(home_dir, 'data')
work_dir = os.path.join(base_dir, 'microscopy-dataset')
train_dir = os.path.join(work_dir, 'training')

image_dir = os.path.join(train_dir, 'images')
mask_dir = os.path.join(train_dir, 'masks')

SIZE = 256
image_dataset = []
mask_dataset = []

images = os.listdir(image_dir)
for i, image_name in tqdm(enumerate(images)):
    if image_name.split('.')[1] == 'tif':
        image = cv2.imread(os.path.join(image_dir, image_name), 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_dir)
for i, image_name in tqdm(enumerate(masks)):
    if image_name.split('.')[1] == 'tif':
        image = cv2.imread(os.path.join(mask_dir, image_name), 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

print(len(image_dataset))
print(len(mask_dataset))