import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('handwritten.keras')

# Counter for image filenames
imgnum = 0

# Iterate through files
while os.path.isfile(f"digits/{imgnum}.png"):
    try:
        # Load the image as grayscale
        img = cv2.imread(f"digits/{imgnum}.png", cv2.IMREAD_GRAYSCALE)
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28))
        
        # Invert and normalize the image
        img = np.invert(img)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        
        # Reshape for prediction (batch_size=1, height=28, width=28)
        img = img.reshape(1, 28, 28)

        # Predict the digit
        pred = model.predict(img)
        print(f"Image {imgnum} is probably: {np.argmax(pred)}")

        # Display the image
        #plt.imshow(img[0], cmap=plt.cm.binary)
        #plt.show()

    except Exception as e:  # Catch and print any errors
        print(f"Error processing image {imgnum}: {e}")

    finally:
        imgnum += 1
