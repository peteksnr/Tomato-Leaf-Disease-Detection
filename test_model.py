import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path to your saved model
model_path = '/path/to/your/saved/model'

# Path to the single image you want to test
image_path = '/path/to/your/image/you/want/to/test'

# Class labels 
class_labels = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Preprocess the image
img = image.load_img(image_path, target_size=(224, 224))  # Resize to match input shape of the model
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize the image

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
predicted_label = class_labels[predicted_class]  # Get the label from class labels

# Display the result of prediction in a popup with a red border and label in top-left
# this section is only for displaying result.
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img) # Display the image
border_color = 'red' # Add red border around the image
border_thickness = 5
rect = patches.Rectangle(
    (0, 0), img.size[0], img.size[1], linewidth=border_thickness, edgecolor=border_color, facecolor='none'
)
ax.add_patch(rect)
# Add predicted class label in top-left corner
ax.text(
    5, 15, predicted_label, color='red', fontsize=12, fontweight='bold', 
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
)
ax.axis('off') # Turn off axes
plt.show() # Show the result of prediction