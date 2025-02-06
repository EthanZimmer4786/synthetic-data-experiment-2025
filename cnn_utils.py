import os
import keras.api
import numpy as np
import random

import matplotlib.pyplot as plt
import visualkeras
from PIL import ImageFont

import tensorflow as tf

import keras
from keras import layers
# from keras.api.models import Sequential

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

########## Batch Size Decider ##########

def calc_batch_size(x):
    step = 0
    while True:
        if x * 0.01 < 2**step:
            return 2**step
        else:
            step += 1

########## Dataset Creation From Directory ##########

def create_dataset(path, subset, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, SEED, LOGGER):
    if subset != False:
        return keras.utils.image_dataset_from_directory(
            path,
            labels = 'inferred',
            label_mode = 'int',
            class_names = None,
            color_mode = 'rgb',
            batch_size = BATCH_SIZE,
            image_size = (IMG_HEIGHT, IMG_WIDTH),
            shuffle = True,
            seed = SEED,
            validation_split = 0.2,
            subset = subset,
            interpolation = 'bilinear',
            follow_links = False,
            crop_to_aspect_ratio = False,
            # pad_to_aspect_ratio = False,
            # data_format = None,
            # verbose = LOGGER #True if LOGGER else False
            )
    else:
        return keras.utils.image_dataset_from_directory(
            path,
            labels = 'inferred',
            label_mode = 'int',
            class_names = None,
            color_mode = 'rgb',
            batch_size = BATCH_SIZE,
            image_size = (IMG_HEIGHT, IMG_WIDTH),
            shuffle = True,
            seed = SEED,
            interpolation = 'bilinear',
            follow_links = False,
            crop_to_aspect_ratio = False,
            # pad_to_aspect_ratio = False,
            # data_format = None,
            # verbose = LOGGER #True if LOGGER else False
            )

########## Model Creation ##########

def get_model(input_shape):
    return keras.Sequential([
        input_shape,
        # Data Augmentation
        layers.Rescaling(1./255), # Normalizes 0-255 rgb values to 0-1 #, input_shape = (img_height, img_width, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        # Layers
        layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.05), # 0.00

        layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.05), # 0.00

        layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.05), # 0.00

        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(0.25), # 0.50
        layers.Dense(10) # Number Of Classes
    ])

# font = ImageFont.truetype("arial.ttf", 32)
# visualkeras.layered_view(model, scale_xy = 20, legend = True, font = font, to_file = 'model_diagram_1.png') # write to disk
# keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

########## Data Collection And Visualization ##########

# train_acc = histacy: ', max(val_acc))

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, train_acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, train_loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()