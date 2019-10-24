from keras.models import load_model
print(11)
from sklearn.metrics import accuracy_score
print(12)
import numpy as np
print(13)
import cv2
from tensorflow.keras import layers, models
import tensorflow as tf
print(1)

model = load_model('my_model.h5')
print(21)
model.summary()

def crop_frame(frame, height, width, density):
    return cv2.resize(frame, (width * density, height * density))

def prepare_test_data(frame):
    test = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    test = crop_frame(test, 33, 1, 50)
    test_letters = np.vsplit(test, 33)
    test_cells = []
    for d in test_letters:
        # d = d.flatten()
        test_cells.append(d)
    return np.array(test_cells, dtype=np.float32)
print(2)
# Test data preparation
test_cells = prepare_test_data("data/test.png")  # test_letters2.BMP
test_cells = test_cells[:, :, :, None]
test_cells_labels = np.arange(33)
# print(test_cells.shape)
print(3)
# Normalization
test_cells = test_cells / 255.0

test_loss, test_acc = model.evaluate(test_cells, test_cells_labels)
print(test_acc, "Accuracy")
result = model.predict(test_cells)
print(np.argmax(result, axis=1))
print("SKlearn_accuracy:", accuracy_score(np.argmax(result, axis=1), test_cells_labels))