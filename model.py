import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#load training and test data and preprocess
train_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/asl-hg/asl_processed/train",
    labels='inferred',
    label_mode='categorical',
    shuffle=True,
    seed=123,
    image_size=(224, 224),
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/asl-hg/asl_processed/test",
    labels='inferred',
    label_mode='categorical',
    shuffle=True,
    seed=123,
    image_size=(224, 224),
)

class_names = train_ds.class_names

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

train_ds_2 = tf.keras.utils.image_dataset_from_directory(
    "datasets/synthetic-asl-dataset/train",
    labels='inferred',
    label_mode='categorical',
    shuffle=True,
    seed=123,
    image_size=(224, 224),
)

test_ds_2 = tf.keras.utils.image_dataset_from_directory(
    "datasets/synthetic-asl-dataset/test",
    labels='inferred',
    label_mode='categorical',
    shuffle=True,
    seed=123,
    image_size=(224, 224),
)

train_ds_2 = train_ds_2.map(lambda x, y: (preprocess_input(x), y))
test_ds_2 = test_ds_2.map(lambda x, y: (preprocess_input(x), y))

#load model
base_model = DenseNet201(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

for layer in base_model.layers[:-1]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

#output is 37 nodes (A-Z and 0-10)
output = Dense(37, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#train
history = model.fit(
    train_ds.concatenate(train_ds_2),
    validation_data=test_ds.concatenate(test_ds_2),
    epochs=70,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)

print("test ds 1")
loss1, acc1 = model.evaluate(test_ds)
print("test ds 2")
loss2, acc2 = model.evaluate(test_ds_2)
print("combined acc should be ", (acc1 * 7200 + acc2 * 3600) / (7200 + 3600))
print("actual combined acc")
loss, acc = model.evaluate(test_ds.concatenate(test_ds_2))

model.save("model.keras")

# Get predictions
test_ds_combined = test_ds.concatenate(test_ds_2)
y_pred = model.predict(test_ds_combined)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = np.concatenate([y for x, y in test_ds_combined], axis=0)
y_true_classes = np.argmax(y_true, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# Because we have 37 classes, we need to decrease the font to read them all
plt.rcParams.update({'font.size': 5})
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
# I think plt.show() only works in Jupyter notebooks
# plt.show()
plt.savefig("cm.png")
