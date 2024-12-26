import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Paths to your data folders (customize these paths to match your local system)
train_dir = '/path/to/your/train/dataset'  # Replace with the path to your training dataset
val_dir = '/path/to/your/validation/dataset'  # Replace with the path to your validation dataset
test_dir = '/path/to/your/test/dataset'  # Replace with the path to your test dataset

# Step 1: Data Preparation
# Normalize pixel values for all datasets to the range [0, 1]
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the datasets using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(112, 112),  # Resize all images to 112x112
    batch_size=32,          # Number of images per batch
    class_mode='categorical' # Use categorical mode for multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(112, 112),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(112, 112),
    batch_size=32,
    class_mode='categorical',
    shuffle=False # Do not shuffle for consistent evaluation
)

# Step 2: Build the CNN Model
# Define the CNN architecture
model = models.Sequential([
    # First convolutional layer with 128 filters
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(112, 112, 3)),
    layers.MaxPooling2D((2, 2)),# Max-pooling to reduce spatial dimensions
    # Second convolutional layer with 64 filters
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Third convolutional layer with 32 filters
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Fourth convolutional layer with 16 filters
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2), # Dropout layer to reduce overfitting
    layers.GlobalAveragePooling2D(), # Global average pooling for feature summarization
    layers.Dense(10, activation='softmax') # Output layer for 10 classes
])

# Step 3: Compile the Model
# Compile the model with the Adam optimizer and categorical cross-entropy loss
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'] # Track accuracy during training
)

# Display a summary of the model architecture
model.summary()

# Step 4: Train the Model
# Train the model using the training and validation datasets
history = model.fit(
    train_generator,
    steps_per_epoch=12000,  # Number of steps per epoch (based on dataset size)
    epochs=350,             # Number of training epochs
    validation_data=val_generator,
    validation_steps=3000   # Number of validation steps (based on dataset size)
)

# Save the trained model
model.save('tomato_leaf_disease_model.keras')

# Step 5: Save Training and Validation Accuracy/Loss Plots as Images
# Extract accuracy and loss values from the training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Generate a range for epochs
epochs = range(1, len(acc) + 1)

# Plot training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r^-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'r^-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Save the accuracy and loss plots
plt.savefig("training_validation_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

# Step 6: Calculate and Save Confusion Matrices with Raw Counts
def save_confusion_matrix(generator, set_name, model):
    """
    Generates and saves a confusion matrix with raw counts for a given dataset.
    """
    y_true = generator.classes # True labels from the dataset
    y_pred = model.predict(generator) # Model predictions
    y_pred_classes = np.argmax(y_pred, axis=1) # Convert predictions to class labels
    
    conf_matrix = confusion_matrix(y_true, y_pred_classes)  # Generate confusion matrix
    
    # Create ConfusionMatrixDisplay with raw counts
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=list(generator.class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)

    # Save the confusion matrix as an image
    plt.gcf().set_size_inches(12, 12)
    plt.title(f"{set_name.capitalize()} Confusion Matrix (Raw Counts)")
    plt.savefig(f"{set_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

# Save confusion matrices for training, validation, and test datasets
save_confusion_matrix(train_generator, "train", model)
save_confusion_matrix(val_generator, "validation", model)
save_confusion_matrix(test_generator, "test", model)

# Step 7: Calculate Precision, Recall, F1-Score for Each Class and Show in Table
# Get true labels and predictions for the test dataset
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate a classification report
class_report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
print("Classification Report:")
print(class_report)

# Calculate weighted precision, recall, and F1-score
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# Print the evaluation metrics to the console
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Display all saved images using OpenCV
cv2.imshow("Training and Validation Metrics", cv2.imread("training_validation_metrics.png"))
cv2.imshow("Training Confusion Matrix", cv2.imread("train_confusion_matrix.png"))
cv2.imshow("Validation Confusion Matrix", cv2.imread("validation_confusion_matrix.png"))
cv2.imshow("Test Confusion Matrix", cv2.imread("test_confusion_matrix.png"))

# Wait until a key is pressed to close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()