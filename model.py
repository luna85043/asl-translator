import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#load training and test data and preprocess
train_ds_1 = tf.keras.utils.image_dataset_from_directory(
    "datasets/asl-hg/asl_processed/train",
    labels='inferred',
    label_mode='categorical',
    shuffle=True,
    seed=123,
    image_size=(224, 224),
)

test_ds_1 = tf.keras.utils.image_dataset_from_directory(
    "datasets/asl-hg/asl_processed/test",
    labels='inferred',
    label_mode='categorical',
    shuffle=False,
    seed=123,
    image_size=(224, 224),
)

class_names = train_ds_1.class_names

train_ds_1 = train_ds_1.map(lambda x, y: (preprocess_input(x), y))
test_ds_1 = test_ds_1.map(lambda x, y: (preprocess_input(x), y))

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
    shuffle=False,
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

#output is 35 nodes (A-Z and 1-9)
output = Dense(35, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#train
history = model.fit(
    train_ds_1.concatenate(train_ds_2),
    validation_data=test_ds_1.concatenate(test_ds_2),
    epochs=50,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
) 
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig("history.png")

loss1, acc1 = model.evaluate(test_ds_1)
print("test ds 1: ", acc1)
loss2, acc2 = model.evaluate(test_ds_2)
print("test ds 2: ", acc2)
print("combined acc should be ", (acc1 * 7200 + acc2 * 3600) / (7200 + 3600))
loss, acc = model.evaluate(test_ds_1.concatenate(test_ds_2))
print("actual combined acc: ", acc)

model.save("model.keras")

# Get predictions
cm_data = test_ds_1.concatenate(test_ds_2)
y_pred = model.predict(cm_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = np.concatenate([y for x, y in cm_data], axis=0)
y_true_classes = np.argmax(y_true, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# Because we have 35 classes, we need to decrease the font to read them all
plt.rcParams.update({'font.size': 5})
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
# I think plt.show() only works in Jupyter notebooks
# plt.show()
plt.savefig("cm.png")
