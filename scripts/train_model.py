# scripts/train_model.py
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from data import load_data  # Assuming you have a function to load the dataset

# Set hyperparameters
EPOCHS = 50
INIT_LR = 1e-1
BS = 128

# Load your data
X_train, Y_train, X_test, Y_test = load_data()

# Build the model
model = build_resnet_model((28, 28, 1))  # Example input shape

# Compile the model
optimizer = SGD(learning_rate=INIT_LR, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint_callback = ModelCheckpoint('models/OCR_ResNet_best.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BS,
    validation_data=(X_test, Y_test),
    callbacks=[checkpoint_callback, early_stop_callback],
    verbose=2
)

# Save the final model
model.save('models/OCR_ResNet.h5', save_format=".h5")
