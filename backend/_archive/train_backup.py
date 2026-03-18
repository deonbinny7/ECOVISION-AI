import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_model():
    # Load MobileNetV2 with pre-trained weights, excluding the top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(6, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    return model

def main():
    base_dir = r"C:\Users\deonb\OneDrive\Desktop\ESE\GARBAGE CLASSIFICATION"
    train_dir = os.path.join(base_dir, 'TRAIN')
    validation_dir = os.path.join(base_dir, 'TEST', 'Garbage classification')
    
    target_size = (224, 224)
    batch_size = 32
    
    # Preprocessing & Advanced Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    print("Loading validation data...")
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    model = create_model()
    model.summary()
    
    # Callbacks for better training
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    print("Starting training (Phase 1: Top Layers)...")
    model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Optional: Unfreeze some layers for fine-tuning
    print("Fine-tuning base model...")
    base_model = model.layers[0]
    base_model.trainable = True
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[early_stop, reduce_lr]
    )

    print("Saving model to model.h5...")
    model.save('model.h5')
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
