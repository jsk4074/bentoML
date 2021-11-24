import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pickle

# Load training data / preprossesing
# (ds_train, ds_test), ds_info = tfds.load(
#     'mnist',
#     split=['train', 'test'],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )

# def normalize_img(image, label):
#   """Normalizes images: `uint8` -> `float32`."""
#   return tf.cast(image, tf.float32) / 255., label

# # training pipeline
# ds_train = ds_train.map(
#     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.batch(128)
# ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# # evaluation pipeline
# ds_test = ds_test.map(
#     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# # Load training data
# iris = datasets.load_iris()
# X, y = iris.data, iris.target

# Model Training
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# model.fit(
#     ds_train,
#     epochs=6,
#     validation_data=ds_test,
# )

# history = model

# model.save('models/basic.h5')
model.load_weights('models/basic.h5')

# import the MnistClassifier class defined above
from mnist_classifier import MnistClassifier

# Create a iris classifier service instance
Mnist_classifier_service = MnistClassifier()

# Pack the newly trained model artifact
Mnist_classifier_service.pack('model', model)

# Save the prediction service to disk for model serving
saved_path = Mnist_classifier_service.save()