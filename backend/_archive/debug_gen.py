import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = r"C:\Users\deonb\OneDrive\Desktop\ESE\GARBAGE CLASSIFICATION"
train_dir = os.path.join(base_dir, 'TRAIN')

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_dir = os.path.join(base_dir, 'TEST')
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

vx, vy = next(validation_generator)
print(f"Validation Y shape: {vy.shape}")
