from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# 📁 Define absolute paths
BASE_DIR = os.path.abspath("/Users/milindverma/Desktop/all_the_thing")
TRAIN_DIR = os.path.join(BASE_DIR, "images", "train")
TEST_DIR = os.path.join(BASE_DIR, "images", "test")

# 📦 Create DataFrame with image paths and labels
def createdataframe(directory):
    image_paths, labels = [], []
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Error: Directory not found -> {directory}")
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for imagename in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, imagename))
                labels.append(label)
            print(f"{label} completed")
    return image_paths, labels

train = pd.DataFrame()
train['image_paths'], train['labels'] = createdataframe(TRAIN_DIR)

test = pd.DataFrame()
test['image_paths'], test['labels'] = createdataframe(TEST_DIR)

print("Train Sample:\n", train.head())
print("Test Sample:\n", test.head())

# 🧠 Feature extraction from grayscale images
def extract_features(images):
    features = []
    for image in tqdm(images, desc="Extracting features"):
        try:
            img = load_img(image, color_mode="grayscale", target_size=(48, 48))
            img = np.array(img)
            features.append(img)
        except Exception as e:
            print(f"Error loading image {image}: {e}")
    features = np.array(features).reshape(len(features), 48, 48, 1)
    return features

x_train = extract_features(train['image_paths']) / 255.0
x_test = extract_features(test['image_paths']) / 255.0

# 🎯 Label encoding and one-hot
le = LabelEncoder()
le.fit(train['labels'])

num_classes = len(le.classes_)
y_train = to_categorical(le.transform(train['labels']), num_classes=num_classes)
y_test = to_categorical(le.transform(test['labels']), num_classes=num_classes)

# 🔁 Advanced Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# 🧱 CNN Model Architecture (Improved)
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 🧪 Compile the model
loss_fn = CategoricalCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# 🛑 Callbacks
early_stop = EarlyStopping(patience=6, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
checkpoint = ModelCheckpoint("best_emotion_model.keras", monitor="val_accuracy", save_best_only=True, mode='max')

# 🚀 Train the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, lr_scheduler, checkpoint]
)

# 💾 Save the final model
model_save_path_h5 = os.path.join(BASE_DIR, "emotion_recognition_model.h5")
model_save_path_keras = os.path.join(BASE_DIR, "emotion_recognition_model.keras")

model.save(model_save_path_h5)
model.save(model_save_path_keras)

print(f"✅ Training complete and model saved at:\n- {model_save_path_h5}\n- {model_save_path_keras}")