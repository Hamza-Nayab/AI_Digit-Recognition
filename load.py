import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)
# y_test = tf.keras.utils.normalize(y_test, axis=1)

model = tf.keras.models.load_model('handwritten.keras')

loss, accuracy = model.evaluate(x_test, y_test)

# Evaluate the model

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")