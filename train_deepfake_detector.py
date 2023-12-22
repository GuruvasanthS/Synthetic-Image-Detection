
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 64
NUM_SAMPLES = 10

# Generate dummy dataset
real_images = np.random.randn(NUM_SAMPLES, IMG_SIZE, IMG_SIZE, 3) * 255
fake_images = [cv2.GaussianBlur((np.random.randn(IMG_SIZE, IMG_SIZE, 3) * 255).astype(np.uint8), (5, 5), 0) for _ in range(NUM_SAMPLES)]
fake_images = np.array(fake_images)

real_labels = np.zeros(NUM_SAMPLES)
fake_labels = np.ones(NUM_SAMPLES)

X = np.vstack([real_images.astype(np.uint8), fake_images])
y = np.hstack([real_labels, fake_labels])

indices = np.arange(2 * NUM_SAMPLES)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('deepfake_detector_model.h5')
