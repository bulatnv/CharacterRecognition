import cv2
import numpy as np


def crop_frame(frame, height, width, density):
    return cv2.resize(frame, (width * density, height * density))


def split_image(frame):
    cs = []
    rows = np.vsplit(frame, 34)
    for row in rows:
        row_cs = np.hsplit(row, 24)
        for c in row_cs:
            # for kNN
            c = c.flatten()
            cs.append(c)
    return cs


def process_frame(frame):
    digits = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    digits = crop_frame(digits, 34, 24, 50)
    return np.array(split_image(digits), dtype=np.float32)


def prepare_test_data(frame):
    test = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    test = crop_frame(test, 34, 1, 50)
    test_letters = np.vsplit(test, 34)
    test_cells = []
    for d in test_letters:
        d = d.flatten()
        test_cells.append(d)
    return np.array(test_cells, dtype=np.float32)


cells = []
cells1 = process_frame("201.png")
cells2 = process_frame("201.png")
cells3 = process_frame("201.png")
cells4 = process_frame("201.png")
cells5 = process_frame("201.png")
cells = np.concatenate((cells1, cells2, cells3, cells4, cells5), axis=0)

# cv2.imshow("cell", cells[0])
# cv2.waitKey()
# print(len(cells))

k = np.arange(34)
cells_labels = np.repeat(k, 24)
cells_labels = np.repeat(cells_labels, 5)
# print(len(cells_labels))

# Test data preparation
test_cells = prepare_test_data("test_letters2.BMP")  # test_letters.png

# KNN
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)

print(result)
