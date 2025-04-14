import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# Load trained model
model = tf.keras.models.load_model("deepfake_detector_final.keras")

# Image size (same as training input)
IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    """Load and preprocess the image."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def predict_image():
    """Handle image selection and prediction."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Preprocess the image
    img = preprocess_image(file_path)

    # Predict using model
    prediction = model.predict(img)[0]
    class_names = ["Real", "Fake"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Display image
    img_pil = Image.open(file_path)
    img_pil = img_pil.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Show result
    result_label.config(text=f"Prediction: {predicted_class} ({confidence:.2f}%)",
                        fg="green" if predicted_class == "Real" else "red")


# Create GUI window
root = tk.Tk()
root.title("Deepfake Detector")
root.geometry("500x500")
root.configure(bg="white")

# Upload button
upload_btn = tk.Button(root, text="Select Image", command=predict_image, font=("Arial", 14), bg="#4CAF50", fg="white")
upload_btn.pack(pady=20)

# Image display label
image_label = tk.Label(root, bg="white")
image_label.pack()

# Prediction result label
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="white")
result_label.pack(pady=10)

# Run application
root.mainloop()