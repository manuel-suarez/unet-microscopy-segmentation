"""
Mitochondria semantic segmentation using U-net, Attention Unet and Att Res Unet
"""

import os

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
print(images.shape)