import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from data_preprocessing import load_data

validation_dir = "/users/jakelee/dataset/validation"
_, validation_generator = load_data(train_dir=None, validation_dir=validation_dir)  # Only load validation

model = load_model("animal_kingdom_cnn_model.h5")

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Detailed metrics
true_labels = validation_generator.classes
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
class_names = list(validation_generator.class_indices.keys())

print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_names))
