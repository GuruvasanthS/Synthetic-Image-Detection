import customtkinter as ctk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Constants
IMG_SIZE = 64

# Load the trained model
model = load_model('deepfake_detector_model.h5')

def predict_deepfake(file_path):
    # Load and preprocess the image
    img = cv2.imread(file_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict using the model
    prediction = model.predict(img)
    print("Raw Prediction:", prediction)  # Add this line for debugging

    if np.argmax(prediction) == 0:
        return "The image is REAL"
    else:
        return "The image is FAKE"


def upload_file():
    # Function to handle file upload
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Predict if the image is a deepfake
        result = predict_deepfake(file_path)
        result_text.set(result)

# Create the main application window
app = ctk.CTk()
app.title("DeepFake Detector")

# Add a label with instructions
#instructions_label = ctk.CTkLabel(app, text="Upload an image to check for deepfakes.")
#instructions_label.pack(pady=20)

instructions_label = ctk.CTkLabel(app, text="Synthetic Image Detection", font=ctk.CTkFont(size=45, weight="bold"),text_color="white")
instructions_label.pack(pady=50)

# Add a button to upload images
upload_button = ctk.CTkButton(app, text="Select File", command=upload_file)
upload_button.pack(pady=60)

# Create a StringVar to display the result
# result_text = ctk.StringVar()
# if result_text in "The image is REAL":
#     result_label = ctk.CTkLabel(app, textvariable=result_text,font=ctk.CTkFont(size=25, weight="bold"),text_color="green")
# result_label = ctk.CTkLabel(app, textvariable=result_text,font=ctk.CTkFont(size=25, weight="bold"),text_color="white")
# result_label.pack(pady=40)

# Create a StringVar to display the result
result_text = ctk.StringVar()

# Add a label to display the result
result_label = ctk.CTkLabel(app, font=ctk.CTkFont(size=25, weight="bold"), text_color="white")
result_label.pack(pady=40)

if result_text.get() == "The image is REAL":
    result_label.configure(textvariable=result_text, font=ctk.CTkFont(size=25, weight="bold"), text_color="green")
else:
    result_label.configure(textvariable=result_text, font=ctk.CTkFont(size=25, weight="bold"), text_color="white")



# Add an exit button
exit_button = ctk.CTkButton(app, text="Exit", command=app.quit)
exit_button.pack(pady=150)

# Run the application
app.mainloop()
