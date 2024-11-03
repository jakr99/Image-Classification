from data_preprocessing import load_data
from model_building import create_model

train_dir = "/users/jakelee/dataset/train"
validation_dir = "/users/jakelee/dataset/validation"
EPOCHS = 10

train_generator, validation_generator = load_data(train_dir, validation_dir)
model = create_model()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model
model.save("animal_kingdom_cnn_model.h5")
