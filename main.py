'''
Notes on improving accuracy: Shapes dataset is a small dataset. It is in out best interest to 
use data augmentation techniques to make sure that the model is trained on enough samples. This especially 
becomes problematic when we put aside a test and a validation set. 

In this project, no hyperparameter search was applied. It would be useful to experiement on 
a variety of hyperparameters and optimizers.

Finally, as my personal computer is an old one, I wasn't able to train the model for a long time, or increase the 
batch size. Having a good data augmentation scheme, traning longer on larger mini batches is important.
'''


import os
import random
import pandas as pd
import numpy as np
import shutil 
import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt

import tensorflow as tf
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

project_path = os.getcwd()
dataset_path = 'shapes-dataset'

def get_img_and_mask_paths(path, img_ids):
    '''
    Returns the absolute file paths for both images and masks
    '''
    img_paths = []
    mask_paths = []

    for img_id in img_ids:
        img_paths.append(os.path.join(path, img_id + '.jpg'))    
        mask_paths.append(os.path.join(path, img_id + '.png'))
    return img_paths, mask_paths
    
def split_data(path, img_ids, test_proportion):
    '''
    Splits data into train and test. Creates subfolders in the data folder for ImageDataGenerator, copies images.
    Returns train and test image ids, and their paths without the file name.
    '''
    img_paths, mask_paths = get_img_and_mask_paths(path, img_ids)
    
    random.Random(seed).shuffle(img_paths)
    random.Random(seed).shuffle(mask_paths)
    
    
    train_img_path = os.path.join(project_path,path,'train_images')
    train_mask_path = os.path.join(project_path,path,'train_masks')
    test_img_path = os.path.join(project_path,path,'test_images')
    test_mask_path = os.path.join(project_path,path,'test_masks')
    
    if not os.path.isdir(train_img_path):
        os.makedirs(train_img_path)
        
    if not os.path.isdir(train_mask_path):
        os.makedirs(train_mask_path)
        
    if not os.path.isdir(test_img_path):
        os.makedirs(test_img_path)
        
    if not os.path.isdir(test_mask_path):
        os.makedirs(test_mask_path)
    
    test_size = int(dataset_size*test_proportion)
    
    train_ids = img_ids[:-test_size]
    test_ids = img_ids[-test_size:]
    
    test_ids_set = set(img_ids[-test_size:])
    
    for img_id in img_ids:
        if img_id not in test_ids_set:
            shutil.copy(os.path.join(project_path,path, img_id + '.jpg'), os.path.join(train_img_path,img_id + '.jpg'))
            shutil.copy(os.path.join(project_path,path, img_id + '.png'), os.path.join(train_mask_path,img_id + '.png'))
        else:
            shutil.copy(os.path.join(project_path,path, img_id + '.jpg'),  os.path.join(test_img_path,img_id + '.jpg'))
            shutil.copy(os.path.join(project_path,path, img_id + '.png'),  os.path.join(test_mask_path,img_id + '.png'))
            
            
    return train_img_path, train_mask_path, test_img_path, test_mask_path, train_ids, test_ids

def load_images(img_ids, path_img, path_mask):
    '''
    Loads images into numpy arrays and returns them
    '''
    X = np.zeros((len(img_ids),) + img_size + (3,), dtype="float32")
    y = np.zeros((len(img_ids),) + img_size + (1,), dtype="uint8")
    img_ids_list = list(img_ids)
    
    for i, img_id in enumerate(img_ids_list):
        img = load_img(os.path.join(path_img, img_id + '.jpg'), target_size=img_size)
        X[i] = img
    
    
    if path_mask:
    
        for j, img_id in enumerate(img_ids_list):
            mask = load_img(os.path.join(path_mask, img_id + '.png'), target_size=img_size, color_mode="grayscale")
            y[j] = np.expand_dims(mask, 2)
    
    return X, y

def get_generators(train_ids, train_img, train_mask):
    '''
    Creates Keras' ImageDataGenerators for images and groundtruth. Applies data augmentation 
    on only training set. Returns generators for train and validation sets.
    '''
    data_gen_args = dict(featurewise_center=True,
                        featurewise_std_normalization=True,
                        rotation_range=90,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2)

    val_size = int(len(train_ids) * 0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(train_img[:-val_size], augment=True, seed=seed)
    mask_datagen.fit(train_mask[:-val_size], augment=True, seed=seed)

    image_generator = image_datagen.flow(train_img[:-val_size],batch_size=batch_size,shuffle=True, seed=seed)
    mask_generator = mask_datagen.flow(train_mask[:-val_size],batch_size=batch_size,shuffle=True, seed=seed)


    train_generator = zip(image_generator, mask_generator)

    image_datagen_val = ImageDataGenerator()
    image_datagen_val.fit(train_img[-val_size:], augment=True, seed=seed)

    val_image_generator = image_datagen_val.flow(train_img[-val_size:],batch_size=batch_size,shuffle=True, seed=seed)

    mask_datagen_val = ImageDataGenerator()
    mask_datagen_val.fit(train_mask[-val_size:], augment=True, seed=seed)

    val_mask_generator = mask_datagen_val.flow(train_mask[-val_size:],batch_size=batch_size,shuffle=True, seed=seed)

    val_generator = zip(val_image_generator, val_mask_generator)

    return train_generator, val_generator

def get_model(img_size, num_classes):
    '''
    Creates the basic UNET model 
    '''
    inputs = keras.Input(shape=img_size + (3,))

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  
    
    for filters in [64, 128, 256]:        
        
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  
        previous_block_activation = x  

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)


        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  
        previous_block_activation = x 

        
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    '''
    Keras' SparseCategoricalCrossentropy has problems with its own Mean IoU metric. This is a replacement
    to prevent the shape mismatch.
    '''
    def __init__(self,
           y_true=None,
           y_pred=None,
           num_classes=None,
           name=None,
           dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def train_model(img_size, num_classes, epochs, train_gen, val_gen):
    '''
    Compiles and trains the model. Returns the recorded history and the model
    '''
    keras.backend.clear_session()

    model = get_model(img_size, num_classes)

    model.compile(optimizer="adam", loss='SparseCategoricalCrossentropy',metrics=[UpdatedMeanIoU(num_classes = num_classes), 'accuracy'])

    callbacks = [
        ModelCheckpoint("shapes.h5", save_best_only=True),
        EarlyStopping(patience=1, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=1, min_lr=0.00001, verbose=1)
    ]

    results = model.fit(train_gen, validation_data=val_gen, validation_steps=10, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
 
    return results, model



def plot_train_val_loss(results):
    '''
    Plots train and validation loss. 
    '''
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()

if __name__ == '__main__':

    img_ids = [img_name.split('.')[0] for img_name in os.listdir(dataset_path) if img_name.endswith(".jpg")]

    img_width, img_height, num_channels = img_to_array(load_img(os.path.join(dataset_path, img_ids[0] + '.jpg'))).shape
    img_size = (img_width, img_height)
    num_classes = 4
    dataset_size = len(img_ids)
    batch_size = 32
    seed = 40
    epochs = 25
    steps_per_epoch= 50

    train_img_path, train_mask_path, test_img_path, test_mask_path, train_ids, test_ids = split_data(dataset_path, img_ids, test_proportion = 0.1)

    train_img, train_mask = load_images(train_ids, train_img_path, train_mask_path)
    test_img, _ = load_images(test_ids, test_img_path, None)

    train_generator, val_generator = get_generators(train_ids, train_img, train_mask)

    history, model = train_model(img_size, num_classes, epochs, train_generator, val_generator)

    plot_train_val_loss(history)
    preds = model.predict(test_img)