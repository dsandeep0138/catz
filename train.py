from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Add, Activation
from keras.layers import ConvLSTM2D, UpSampling3D, MaxPooling3D, Reshape, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras import regularizers
from keras.optimizers import Adam
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer

run = wandb.init(project='catz')
config = run.config

config.num_epochs = 10
config.batch_size = 32
config.timesteps = 5
config.img_dir = "images"
config.height = 96
config.width = 96

val_dir = 'catz/test'
train_dir = 'catz/train'

augmentation_generator = ImageDataGenerator(rotation_range=30,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='nearest')

# automatically get the data if it doesn't exist
if not os.path.exists("catz"):
    print("Downloading catz dataset...")
    subprocess.check_output(
        "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz", shell=True)


class ImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            val_generator(15, val_dir))
        output = self.model.predict(validation_X)
        wandb.log({
            "input1": [wandb.Image(c[0]) for c in validation_X],
            "input2": [wandb.Image(c[1]) for c in validation_X],
            "input3": [wandb.Image(c[2]) for c in validation_X],
            "input4": [wandb.Image(c[3]) for c in validation_X],
            "input5": [wandb.Image(c[4]) for c in validation_X],
            "output": [wandb.Image(np.concatenate([validation_y[i], o], axis=1)) for i, o in enumerate(output)]
        }, commit=False)

def aug_train_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, config.timesteps, config.width, config.height, 3))
        output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0

        #Image Augmentation by transforming the image using the
        #augmentation features that are defined
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            output_images[i] = np.array(Image.open(cat_dirs[counter + i] + "/cat_result.jpg"))

            params = augmentation_generator.get_random_transform(np.asarray(Image.open(input_imgs[0])).shape)
            imgs = [augmentation_generator.apply_transform(augmentation_generator.standardize(np.asarray(Image.open(img))), params) for img in sorted(input_imgs)]
            for j in range(len(imgs)):
                input_images[i][j] = imgs[j]
            output_images[i] = augmentation_generator.apply_transform(augmentation_generator.standardize(output_images[i]), params)
        yield (input_images, output_images)

        counter += batch_size

def val_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, config.timesteps, config.width, config.height, 3))
        output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            for j in range(len(imgs)):
                input_images[i][j] = np.array(imgs[j])
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg"))
        yield (input_images, output_images)

        counter += batch_size

class ResidualUnit(Layer):
    def __init__(self, **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)

    def call(self, x):
        firstLayer = x
        out = ConvLSTM2D(32, (3, 3), activation='relu', padding='same',
                     return_sequences=True)(x)
        out = MaxPooling3D(pool_size=(1, 2, 2))(out)
        out = ConvLSTM2D(32, (3, 3), activation='relu', padding='same',
                     return_sequences=True)(out)
        out = UpSampling3D((2, 2))(out)
        out = Conv2D(3, (3, 3), activation='relu', padding='same')(out)
        residual = Add()([out, firstLayer])
        return [out, residual]


model = Sequential()
model.add(ConvLSTM2D(32, (3, 3), activation='relu', padding='same',
                 return_sequences=True,
                 input_shape=(5, config.height, config.width, 3)))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(ConvLSTM2D(32, (3, 3), activation='relu', padding='same',
                 return_sequences=False))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.summary()


def perceptual_distance(y_true, y_pred):
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


model.compile(optimizer='adam', loss='logcosh', metrics=[perceptual_distance])

model.fit_generator(aug_train_generator(config.batch_size, train_dir),
                    steps_per_epoch=len(
                        glob.glob(train_dir + "/*")) // config.batch_size,
                    epochs=config.num_epochs, callbacks=[
    ImageCallback(), WandbCallback()],
    validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
    validation_data=val_generator(config.batch_size, val_dir))
