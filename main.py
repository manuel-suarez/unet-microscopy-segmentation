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
data_dir = os.path.join(work_dir, 'segmentation')
results_dir = os.path.join(work_dir, 'results')
figures_dir = os.path.join(results_dir, 'figures')
weights_dir = os.path.join(results_dir, 'weights')
metrics_dir = os.path.join(results_dir, 'metrics')
plots_dir = os.path.join(results_dir, 'plots')
for path in [figures_dir, weights_dir, metrics_dir, plots_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

SIZE = 256

# Use image generators to load images from disk
seed = 24
batch_size = 8
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(rescale = 1/255.,
                         rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')
mask_data_gen_args = dict(rescale = 1/255.,
                          rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='relect',
                          preprocessing_function=lambda x: np.where(x>0, 1, 0).astype(x.dtype))
image_data_generator = ImageDataGenerator(**img_data_gen_args)
num_train_imgs = len(os.listdir(os.path.join(data_dir,'train_images','train')))
image_generator = image_data_generator.flow_from_directory(os.path.join(data_dir,'train_images'),
                                                           seed=seed,
                                                           batch_size=batch_size,
                                                           class_mode=None) # Binary
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow_from_directory(os.path.join(data_dir,'train_masks'),
                                                         seed=seed,
                                                         batch_size=batch_size,
                                                         color_mode='grayscale',
                                                         class_mode=None) # Binary

valid_img_generator = image_data_generator.flow_from_directory(os.path.join(data_dir,'val_images'),
                                                               seed=seed,
                                                               batch_size=batch_size,
                                                               class_mode=None)
valid_mask_generator = mask_data_generator.flow_from_directory(os.path.join(data_dir,'val_masks'),
                                                               seed=seed,
                                                               batch_size=batch_size,
                                                               color_mode='grayscale',
                                                               class_mode=None)
test_img_generator = image_data_generator.flow_from_directory(os.path.join(data_dir,'test_images'),
                                                              seed=seed,
                                                              batch_size=32,
                                                              class_mode=None)
test_mask_generator = mask_data_generator.flow_from_directory(os.path.join(data_dir,'test_masks'),
                                                              seed=seed,
                                                              batch_size=32,
                                                              color_mode='grayscale',
                                                              class_mode=None)
train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)

x = image_generator.next()
y = mask_generator.next()
for i in range(0,1):
    image = x[i]
    mask = y[i]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.savefig(os.path.join(figures_dir,'figure01.png'))
    plt.close()

# Parameters for model
IMG_HEIGHT = x.shape[1]
IMG_WIDTH = x.shape[2]
IMG_CHANNELS = x.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
num_epochs = 50

# Use library implementation
from focal_loss import BinaryFocalLoss

#from models.functional.unet import UNet
from models.implementations.models_v1 import UNet, Attention_UNet, Attention_ResUNet, dice_coef, jacard_coef
from models.functional.unet import UNet as FUnet

def train_model(model, optimizer, loss, metrics, epochs, model_name):
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    start1 = datetime.now()
    model_history = model.fit(train_generator,
                              verbose=1,
                              batch_size=batch_size,
                              steps_per_epoch=num_train_imgs//batch_size,
                              validation_data=val_generator,
                              validation_steps=num_train_imgs//batch_size,
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
                    epochs=50,
                    model_name=name)