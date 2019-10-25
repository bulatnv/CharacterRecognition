# import the necessary packages
import os
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2


def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def crop_by_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break

    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    return four_point_transform(frame, docCnt.reshape(4, 2))


def rename_imgs():
    dir_path = 'dataset_raw'
    img_names = os.listdir(dir_path)
    for old_name, i in zip(img_names, range(len(img_names))):
        os.rename(os.path.join(dir_path, old_name), os.path.join(dir_path, '{0}.bmp'.format(i)))


def proccess_all_images_in_folder():
    rename_imgs()
    filePath = "dataset_raw/"
    fileType = ".bmp"

    for i in range(77):
        file = filePath + str(i) + fileType
        image = cv2.imread(file)

        paper = crop_by_edges(image)
        grayImage = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(grayImage, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cv2.imwrite(filePath + "processed" + str(i) + fileType, thresh)


def proccess_image_in_folder(name):
    filePath = "dataset_raw/"
    fileType = ".bmp"

    file = filePath + str(name) + fileType
    print(file)
    image = cv2.imread(file)

    paper = crop_by_edges(image)
    grayImage = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(grayImage, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cv2.imwrite(filePath + "processed" + str(name) + fileType, thresh)


proccess_all_images_in_folder()
