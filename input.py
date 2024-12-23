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
        # Load the image as grayscale and resize to 28x28
        img = cv2.imread(f"digits/{imgnum}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        
        # Invert the colors (if necessary)
        img = np.invert(np.array([img]))
        pred = model.predict(img)
        print(f"img is prolly {np.argmax(pred)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()

    except Exception as e:  # Catch and print any errors
        print(f"Error processing image {imgnum}: {e}")

    finally:
        
        imgnum += 1
        
