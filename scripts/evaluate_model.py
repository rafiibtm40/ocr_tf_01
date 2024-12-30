# scripts/evaluate_model.py
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data import load_data  # Assuming you have a function to load the dataset

# Load data
X_train, Y_train, X_test, Y_test = load_data()

# Load the saved model
model = load_model('models/OCR_ResNet_best.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Plotting training history (assuming `history` object is available)
# If you have the history saved, you can load it or re-run training to get it.

plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
