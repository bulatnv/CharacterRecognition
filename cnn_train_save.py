from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from create_dataset import load_dataset, prepare_test_data

# Creating labels
k_labels = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У',
            'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ь', 'Ы', 'Ъ', 'Э', 'Ю', 'Я']

# processing images
print("Loading images and labels ...")
cells, labels = load_dataset()
cells = np.array(cells)
cells = cells[:, :, :, None]
print("Images and labels loaded!!! Done! \n")

# Test data preparation
print("Prepearing test data ...")
test_cells = prepare_test_data("data/test.png")  # test_letters2.BMP
test_cells = test_cells[:, :, :, None]
test_cells_labels = np.arange(33)
print("Done!!\n")


print("Prepearing for training ...")
# Normalization
cells, test_cells = cells / 255.0, test_cells / 255.0

# CNN
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))

model1.add(layers.Flatten())
model1.add(layers.Dense(80, activation='relu'))
model1.add(layers.Dense(34, activation='softmax'))
model1.summary()

model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model1.fit(cells, labels, shuffle=True, epochs=5)
print("Training model. Done!!! \n\n")

print("Saving model ...")
model1.save('my_model.h5')

test_loss, test_acc = model1.evaluate(test_cells, test_cells_labels)
print(test_acc, "Accuracy")
result = model1.predict(test_cells)
print(np.argmax(result, axis=1))
print("SKlearn_accuracy:", accuracy_score(np.argmax(result, axis=1), test_cells_labels))
