"""
Mitochondria semantic segmentation using U-net, Attention Unet and Att Res Unet
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam

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

# Normalize images
image_dataset = np.array(image_dataset)/255.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3)/255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

# Parameters for model
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
batch_size = 8

# Use library implementation
from focal_loss import BinaryFocalLoss

from models import UNet, dice_coef, dice_loss, IoU_coef

unet_model = UNet(input_shape)
unet_model.compile(optimizer=Adam(lr = 1e-2), loss=BinaryFocalLoss(gamma=2),
                   metrics=['accuracy', IoU_coef])
print(unet_model.summary())

start1 = datetime.now()
unet_history = unet_model.fit(X_train, y_train,
                              verbose=1,
                              batch_size=batch_size,
                              validation_data=(X_test, y_test),
                              shuffle=False,
                              epochs=50)
stop1 = datetime.now()
# Execution time of the model
execution_time_Unet = stop1-start1
print("UNet execution time is: ", execution_time_Unet)

unet_model.save('mitochondria_UNet_50epochs_B_focal.hdf5')