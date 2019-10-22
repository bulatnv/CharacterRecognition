from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
import cv2
import numpy as np


def crop_frame(frame, height, width, density):
    return cv2.resize(frame, (width * density, height * density))


def split_image(frame):
    cs = []
    rows = np.vsplit(frame, 33)
    for row in rows:
        row_cs = np.hsplit(row, 25)
        for c in row_cs:
            # for kNN
            # c = c.flatten()
            cs.append(c)
    return cs


def process_frame(frame):
    digits = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    digits = crop_frame(digits, 33, 25, 50)
    return np.array(split_image(digits), dtype=np.float32)


def prepare_test_data(frame):
    test = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    test = crop_frame(test, 33, 1, 50)
    test_letters = np.vsplit(test, 33)
    test_cells = []
    for d in test_letters:
        # d = d.flatten()
        test_cells.append(d)
    return np.array(test_cells, dtype=np.float32)


# processing images
cells = []
cells1 = process_frame("data/201.png")
cells2 = process_frame("data/201.png")
cells3 = process_frame("data/201.png")
cells4 = process_frame("data/201.png")
cells5 = process_frame("data/201.png")
cells = np.concatenate((cells1, cells2, cells3, cells4, cells5), axis=0)
cells = cells[:, :, :, None]
# print(cells.shape)

# cv2.imshow("cell", cells[0])
# cv2.waitKey()
# print(len(cells))

# Creating labels
k_labels = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У',
            'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ь', 'Ы', 'Ъ', 'Э', 'Ю', 'Я']
k = range(33)
cells_labels = np.repeat(k, 25)
labels = np.concatenate((cells_labels, cells_labels, cells_labels, cells_labels, cells_labels), axis=None)


# Test data preparation
test_cells = prepare_test_data("data/test.png")  # test_letters2.BMP
test_cells = test_cells[:, :, :, None]
test_cells_labels = np.arange(33)
# print(test_cells.shape)

## Normalization
cells, test_cells = cells / 255.0, test_cells / 255.0

#CNN
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
model1.fit(cells, labels, shuffle=True, epochs=8)

test_loss, test_acc = model1.evaluate(test_cells, test_cells_labels)

print(test_acc, "Accuracy")
result = model1.predict(test_cells)
print(np.argmax(result, axis=1))
print("SKlearn_accuracy:", accuracy_score(np.argmax(result, axis=1), test_cells_labels))
