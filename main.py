"""
Mitochondria semantic segmentation using U-net, Attention Unet and Att Res Unet
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pandas import DataFrame
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam

home_dir = os.path.expanduser('~')
base_dir = os.path.join(home_dir, 'data')
work_dir = os.path.join(base_dir, 'microscopy-dataset')
train_dir = os.path.join(work_dir, 'training')
results_dir = os.path.join(work_dir, 'results')

image_dir = os.path.join(train_dir, 'images')
mask_dir = os.path.join(train_dir, 'masks')
figures_dir = os.path.join(results_dir, 'figures')
weights_dir = os.path.join(results_dir, 'weights')
metrics_dir = os.path.join(results_dir, 'metrics')
plots_dir = os.path.join(results_dir, 'plots')
for path in [figures_dir, weights_dir, metrics_dir, plots_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

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
num_epochs = 5

# Use library implementation
from focal_loss import BinaryFocalLoss

#from models.functional.unet import UNet
from models.implementations.models_v1 import UNet, Attention_UNet, Attention_ResUNet, dice_coef, jacard_coef
from models.functional.unet import UNet as FUnet

def train_model(model, optimizer, loss, metrics, epochs, model_name):
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    start1 = datetime.now()
    model_history = model.fit(X_train, y_train,
                                  verbose=1,
                                  batch_size=batch_size,
                                  validation_data=(X_test, y_test),
                                  shuffle=False,
                                  epochs=epochs)
    stop1 = datetime.now()
    # Execution time of the model
    execution_time_Unet = stop1 - start1
    print(f"{model_name} execution time is: ", execution_time_Unet)

    # Save model
    fname = '-'.join(model_name.split(' '))
    model.save(os.path.join(weights_dir, f"mitochondria_{fname}_50epochs_B_focal.hdf5"))
    # Save history
    model_history_df = DataFrame(model_history.history)
    with open(os.path.join(metrics_dir, f"{fname}_history_df.csv"), mode='w') as f:
        model_history_df.to_csv(f)
    # Plot training loss and metrics
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{fname}_loss.png"))
    plt.close()

    # acc = history.history['jacard_coef']
    acc = model_history.history['accuracy']
    # val_acc = history.history['val_jacard_coef']
    val_acc = model_history.history['val_accuracy']

    plt.figure()
    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Jacard')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{fname}_accuracy.png"))
    plt.close()

    # Save segmentation results
    # Load one model at a time for testing.
    model_path = os.path.join(weights_dir, f"mitochondria_{fname}_50epochs_B_focal.hdf5")
    model = tf.keras.models.load_model(model_path, compile=False)

    import random

    test_img_number = random.randint(0, X_test.shape[0] - 1)
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]

    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img, cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')

    plt.savefig(os.path.join(figures_dir, f"{fname}_prediction.png"))
    plt.close()

    # IoU for a single image
    from tensorflow.keras.metrics import MeanIoU

    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(ground_truth[:, :, 0], prediction)
    print("Mean IoU =", IOU_keras.result().numpy())

    # Calculate IoU for all test images and average
    import pandas as pd

    IoU_values = []
    for img in range(0, X_test.shape[0]):
        temp_img = X_test[img]
        ground_truth = y_test[img]
        temp_img_input = np.expand_dims(temp_img, 0)
        prediction = (model.predict(temp_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

        IoU = MeanIoU(num_classes=n_classes)
        IoU.update_state(ground_truth[:, :, 0], prediction)
        IoU = IoU.result().numpy()
        IoU_values.append(IoU)

        print(IoU)

    df = pd.DataFrame(IoU_values, columns=["IoU"])
    df = df[df.IoU != 1.0]
    mean_IoU = df.mean().values
    print("Mean IoU is: ", mean_IoU)

if __name__ == '__main__':
    unet_model = UNet(input_shape)
    funet_model = FUnet(input_shape)
    #att_unet_model = Attention_UNet(input_shape)
    #att_res_unet_model = Attention_ResUNet(input_shape)
    models = [unet_model, funet_model]
    names = ['UNet', 'FUNet']
    for model, name in zip(models, names):
        train_model(model,
                    optimizer=Adam(learning_rate=1e-2),
                    loss=BinaryFocalLoss(gamma=2),
                    metrics=['accuracy', dice_coef, jacard_coef],
                    epochs=5,
                    model_name=name)