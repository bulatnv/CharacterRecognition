# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
from cnn_load_inference import process_letter_field, process_digit_field


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


def crop_by_lines(frame):
    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL,
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


def crop_questions(frame, y1, y2, x1, x2, debug=False):
    hight, width = frame.shape[:2]
    xDim = 176 / width
    yDim = 256 / hight
    x1, x2 = int(x1 / xDim), int(x2 / xDim)
    y1, y2 = int(y1 / yDim), int(y2 / yDim)
    if debug:
        print(frame.shape)
        print(y1, y2, x1, x2)
        cv2.imshow('frame_test', frame[y1:y2, x1:x2])
        cv2.waitKey()
    return frame[y1:y2, x1:x2]


def crop_input_field(frame, x1, x2, y1, y2, debug=False):
    return crop_by_lines(crop_questions(frame, x1, x2, y1, y2, debug))


def check_bubbles(thresh):
    # cv2.imshow("quiz", thresh)
    # cv2.waitKey()
    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
            questionCnts.append(c)

    # sort the question contours top-to-bottom, then initialize
    # the total number of correct answers
    questionCnts = contours.sort_contours(questionCnts,
                                          method="top-to-bottom")[0]

    Keys = {}
    ans = []
    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        ans = []
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            # cv2.imshow("maska", mask)
            # cv2.waitKey()
            total = cv2.countNonZero(mask)
            # print(total)

            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            if total > 4200:
                ans.append(j)
        Keys[q] = ''.join(str(i) for i in ans)
    return Keys


def core(str):
    image = cv2.imread(str)

    paper_color = crop_by_edges(image)
    cv2.imwrite('debug/paper_color.png', paper_color)
    paper_gray = cv2.cvtColor(paper_color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('debug/paper_gray.png', paper_gray)
    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    black = cv2.threshold(paper_gray, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imwrite('debug/black.png', black)

    # reading names
    test_id = crop_input_field(black, 2, 14, 60, 120)
    # cv2.imshow('test_id', test_id)
    classnum = crop_input_field(black, 2, 14, 138, 170)
    name = crop_input_field(black, 16, 26, 36, 170)
    surname = crop_input_field(black, 28, 38, 36, 170)
    answers = crop_input_field(black, 40, 240, 17, 170)
    cv2.imwrite('debug/name.png', name)
    cv2.imwrite('debug/surname.png', surname)

    test_id = process_digit_field(test_id, 8)
    name = process_letter_field(name, 20)
    surname = process_letter_field(surname, 20)
    print(test_id, name, surname)

    # reading questions
    # answers = check_bubbles(crop_questions(black, 0, 90, 50, 290))
    # other_answers1 = check_bubbles(crop_questions(black, 120, 200, 50, 290))
    # other_answers1 = other_answers1.items()
    # i = len(answers)
    # for j in other_answers1:
    #     answers[i] = j[1]
    #     i += 1
    return name, surname


N, S = core('scan/7.png')
# print(N, S, F)
# print(KEYS)
