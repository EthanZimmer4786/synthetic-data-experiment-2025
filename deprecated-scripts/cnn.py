import os
import numpy as np
import random

import matplotlib.pyplot as plt
import visualkeras
from PIL import ImageFont

import tensorflow as tf

import keras
from keras import layers
from keras.api.models import Sequential

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

AUTOTUNE = tf.data.AUTOTUNE

# Dataset
dataset_name = 'Dataset 2024-12-13 11.12.09'
train_ds_path = './Generated Datasets/' + dataset_name + '/train/'
test_path = './Generated Datasets/' + dataset_name + '/test/real/'

batch_size = 64
img_height = 32
img_width  = 32

seed = round(random.random() * 100)

# Training
epochs = 30 # 50

########## Dataset Creation From Directory ##########

def create_dataset(path, subset):
    return keras.utils.image_dataset_from_directory(
        path,
        labels = 'inferred',
        label_mode = 'int',
        class_names = None,
        color_mode = 'rgb',
        batch_size = batch_size,
        image_size = (img_height, img_width),
        shuffle = True,
        seed = seed,
        validation_split = 0.2,
        subset = subset,
        interpolation = 'bilinear',
        follow_links = False,
        crop_to_aspect_ratio = False,
        pad_to_aspect_ratio = False,
        data_format = None,
        verbose = True
        )

train_ds = create_dataset(train_ds_path, 'training')
val_ds = create_dataset(train_ds_path, 'validation')

print('----------')
train_count = tf.data.experimental.cardinality(train_ds).numpy()
print('Train Dataset Contains: ', train_count, ' Batches of ', batch_size, ' Images')
val_count = tf.data.experimental.cardinality(val_ds).numpy()
print('Validate Dataset Contains: ', val_count, ' Batches of ', batch_size, ' Images')
print('----------')

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

########## Model Creation ##########

data_augmentation = keras.Sequential(
    [
    layers.Rescaling(1./255), # Normalizes 0-255 rgb values to 0-1 #, input_shape = (img_height, img_width, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    ]
)

model = Sequential([
    layers.Input(shape = (img_height, img_width, 3)),
    data_augmentation,

    # layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
    # layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.25),
    layers.Dense(10) # Number Of Classes
])

model.compile(optimizer = keras.optimizers.Adam(),
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

model.summary()

# font = ImageFont.truetype("arial.ttf", 32)
# visualkeras.layered_view(model, scale_xy = 20, legend = True, font = font, to_file = 'model_diagram_1.png') # write to disk
# keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

########## Model Fitting ##########

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs,
    verbose = 1,
)

########## Model Testing ##########
test_ds = keras.utils.image_dataset_from_directory(
    test_path,
    labels = 'inferred',
    label_mode = 'int',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = seed,
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False,
    pad_to_aspect_ratio = False,
    data_format = None,
    verbose = True
    )

results = model.evaluate(test_ds)
print("Loss:", results[0])
print("Accuracy:", results[1])

########## Data Collection And Visualization ##########

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

# print('Average validation accuracy: ', np.average(val_acc))
print("Train Accuracy:", train_acc[-1])
print("Validate Accuracy:", val_acc[-1])

print("Train Loss:", train_loss[-1])
print("Validate Loss:", val_loss[-1])

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