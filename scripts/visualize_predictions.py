import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scripts.load_data import load_data  # Make sure to import from the correct path

# Load the model (make sure the model file path is correct)
model = load_model('models/OCR_ResNet_best.h5')

# Load data (load_data() returns X_train, Y_train, X_test, Y_test)
X_train, Y_train, X_test, Y_test = load_data()

# Predict the labels of a few random images from the test set
predictions = model.predict(X_test[:5])

# Display the images and their predicted vs actual labels
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')  # Ensure image is 28x28 for display
    axes[i].set_title(f"Pred: {np.argmax(predictions[i])}, True: {np.argmax(Y_test[i])}")
    axes[i].axis('off')

plt.show()
