from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')
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


# Test data preparation
test_cells = prepare_test_data("data/test.png")  # test_letters2.BMP
test_cells = test_cells[:, :, :, None]
test_cells_labels = np.arange(33)
# print(test_cells.shape)

# Normalization
test_cells = test_cells / 255.0

test_loss, test_acc = model.evaluate(test_cells, test_cells_labels)
print(test_acc, "Accuracy")
result = model.predict(test_cells)
print(np.argmax(result, axis=1))
print("SKlearn_accuracy:", accuracy_score(np.argmax(result, axis=1), test_cells_labels))
