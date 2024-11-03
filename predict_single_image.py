import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("animal_kingdom_cnn_model.h5")
class_names = ["Animalia", "Plantae", "Fungi"]  # Adjust based on your classes

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    print(f"Predicted Class: {predicted_class}")

# Example usage
predict_image("/users/jakelee/dataset/lion.jpg")
