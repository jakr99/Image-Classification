from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir=None, validation_dir=None, img_size=(128, 128), batch_size=32):
    train_generator = None
    validation_generator = None
    
    # Only create the train generator if train_dir is provided
    if train_dir:
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
    
    # Only create the validation generator if validation_dir is provided
    if validation_dir:
        validation_datagen = ImageDataGenerator(rescale=1.0/255)
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
    
    return train_generator, validation_generator
