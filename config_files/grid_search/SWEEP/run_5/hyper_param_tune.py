#!/usr/bin/env python
# coding: utf-8

# In[]:

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# In[2]: Load MNIST data

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# In[]: Normalize training features to values within [0, 1]

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


# In[]: Normalize testing features to values within [0, 1]

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# In[]: Train an ANN

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

# In[] Store training and test accuracy

_, accuracy_train = model.evaluate(ds_train)
_, accuracy_test = model.evaluate(ds_test)

np.savetxt('output.csv', np.array([accuracy_train, accuracy_test]).reshape([1, 2]),
           header = "accuracy_train,accuracy_test", delimiter=",",
           comments="")
