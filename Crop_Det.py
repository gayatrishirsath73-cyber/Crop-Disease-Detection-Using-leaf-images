#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# Extract Dataset (FIXED PATH)
# -----------------------------
import zipfile, os

zip_path = "Dataset PlantVillage.zip"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

# -----------------------------
# DATA DIRECTORY
# -----------------------------
data_dir = "Dataset PlantVillage"

# Safety check (prevents crash)
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"{data_dir} not found")

print("Final Path:", data_dir)
print(os.listdir(data_dir)[:5])

# -----------------------------
# LOAD DATASET
# -----------------------------
import tensorflow as tf

img_size = (128,128)
batch_size = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# -----------------------------
# Class Distribution
# -----------------------------
import matplotlib.pyplot as plt

class_count = []

for i in class_names:
    count = len(os.listdir(os.path.join(data_dir, i)))
    class_count.append(count)

plt.bar(class_names, class_count)
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=90)
plt.show()

# -----------------------------
# DATA ANALYSIS
# -----------------------------
plt.figure(figsize=(8,5))
for images, labels in train_ds.take(1):
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# -----------------------------
# NORMALIZATION
# -----------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# -----------------------------
# CNN Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(128,128,3)),
    data_augmentation,

    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# PERFORMANCE OPTIMIZATION
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# TRAIN (FASTER)
# -----------------------------
print("Starting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,     # reduced epochs for speed
    verbose=1
)

print("Training completed")

model.save("model.h5")
print("Model saved successfully")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
from sklearn.metrics import confusion_matrix
import numpy as np

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=90)
plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.colorbar()
plt.show()

# -----------------------------
# SIMPLE IMAGE TEST
# -----------------------------
from PIL import Image

img_path = "tomato leaf2.jpg"

if os.path.exists(img_path):
    img = Image.open(img_path).resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]

    print("Predicted Class:", predicted_class)

    if "healthy" in predicted_class.lower():
        print("Result: Healthy Leaf")
    else:
        print("Result: Diseased Leaf")

    plt.imshow(img)
    plt.title("Prediction: " + predicted_class)
    plt.axis("off")
    plt.show()