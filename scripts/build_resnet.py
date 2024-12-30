from tensorflow.keras import layers, models

def build_resnet_model(input_shape):
    # Start the model
    input_layer = layers.Input(shape=input_shape)

    # First Convolution Block
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual Block 1
    shortcut = x
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])  # Adding residual connection

    # Residual Block 2
    shortcut = x
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])  # Adding residual connection

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense Layer for classification
    output_layer = layers.Dense(26, activation="softmax")(x)  # 26 characters for A-Z classification

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

if __name__ == "__main__":
    # Example: Build ResNet Model for 28x28 grayscale images (MNIST-like dataset)
    input_shape = (28, 28, 1)  # Update this based on your dataset dimensions
    model = build_resnet_model(input_shape)
    model.summary()
