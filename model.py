import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory

#load training and test data and preprocess
train_ds = tf.keras.utils.image_dataset_from_directory(
    "asl_processed/train",
    labels='inferred',
    label_mode='int',
    shuffle=True,
    seed=123,
    image_size=(100, 100),
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "asl_processed/test",
    labels='inferred',
    label_mode='int',
    shuffle=True,
    seed=123,
    image_size=(100, 100),
)

#to do: label images, load into x and y test/train

#load model
base_model = DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

#output is 36 nodes
output = Dense(36, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#train
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32
)

loss, acc = model.evaluate(x_test, y_test)
print("Accuracy:", acc)
