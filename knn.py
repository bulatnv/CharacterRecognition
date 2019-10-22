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
            c = c.flatten()
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


# print(len(cells))

k_labels = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ь', 'Ы', 'Ъ', 'Э', 'Ю', 'Я']
k = range(33)
cells_labels = np.repeat(k, 25)
labels = np.concatenate((cells_labels, cells_labels, cells_labels, cells_labels, cells_labels), axis=None)
# print(len(cells))
# print(len(labels))

# for index in range(0, 4080, 27):
#     print("label: ", k_labels[labels[index]])
#     cv2.imshow("cell", cells[index])
#     cv2.waitKey()

# Test data preparation
test_cells = prepare_test_data("test.png")

# KNN
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)

print(result)
