# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the MNIST dataset
print("Loading the MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Preprocessing
print("Preprocessing the data...")
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Reshape and normalize training data
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0    # Reshape and normalize testing data

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)  # 10 classes (0-9)
y_test = to_categorical(y_test, 10)

# Visualize some examples from the dataset
def plot_sample_images(x, y):
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {np.argmax(y[i])}")
        plt.axis('off')
    plt.show()

print("Displaying sample images from the dataset...")
plot_sample_images(x_train, y_train)

# Define the CNN model
print("Building the CNN model...")
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),  # Adding dropout to prevent overfitting
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Output layer with 10 classes
])

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
print("Model Summary:")
model.summary()

# Train the model
print("Training the model...")
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Plot training & validation accuracy values
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

print("Displaying training and validation history...")
plot_training_history(history)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed evaluation with classification report
print("Generating detailed classification report...")
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion matrix
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

print("Displaying confusion matrix...")
plot_confusion_matrix(conf_matrix, classes=[i for i in range(10)])

# Save the model
model_save_path = "mnist_cnn_model.h5"
print(f"Saving the model to {model_save_path}...")
model.save(model_save_path)

print("Model training and evaluation completed successfully!")
