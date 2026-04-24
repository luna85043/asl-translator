import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#load training and test data and preprocess
train_ds = tf.keras.utils.image_dataset_from_directory(
    "asl_processed/train",
    labels='inferred',
    label_mode='categorical',
    shuffle=True,
    seed=123,
    image_size=(224, 224),
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "asl_processed/test",
    labels='inferred',
    label_mode='categorical',
    shuffle=True,
    seed=123,
    image_size=(224, 224),
)

#to do: label images, load into x and y test/train
class_names = train_ds.class_names

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

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
    train_ds,
    validation_data=test_ds,
    epochs=3
)
# Get predictions
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true_classes = np.argmax(y_true, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

loss, acc = model.evaluate(test_ds)
print("Accuracy:", acc)
