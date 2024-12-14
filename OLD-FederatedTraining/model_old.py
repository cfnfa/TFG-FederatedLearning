import tensorflow as tf


# Verifica si 'tf.keras' está disponible
try:
    model = tf.keras.Sequential()
    print("Keras está correctamente integrado en TensorFlow.")
except AttributeError:
    print("Keras no está disponible en esta instalación de TensorFlow.")

