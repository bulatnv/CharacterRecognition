# Character Recontition lib

проект по распознаванию русских больших букв для
распознавания ответников

### Основная идея:
Собрать библиотеку полезных функций для распознавания 
больших русских букв (А,Б,В,Г,Д,Е,Ё,Ж,З ... Э,Ю,Я)

### Стек технологий:

Для этого предполагается использование сл стек технологий:

* Linux (Ubuntu)
* Python, pip, imutils, numpy, scipy и т.д.
* OpenCV (для определения результатов)
* TensorFlow, Keras

### origin.py

Это оригинал кода по распознаванию цифр

Handwritten digits recognition

https://pysource.com/2018/08/26/knn-handwritten-digits-recognition-opencv-3-4-with-python-3-tutorial-36/

### convert.py

Это код для обрезки листа по черным рамкам

Основную работу выполняет функция crop_by_edges
находит наибольшие рамки и по ним производит обрезку листа.

Handwritten digits recognition: https://pysource.com/2018/08/26/knn-handwritten-digits-recognition-opencv-3-4-with-python-3-tutorial-36/

### convert.py

Это код для обрезки листа по черным рамкам. 
Основную работу выполняет функция crop_by_edges. 
Находит наибольшие рамки и по ним производит обрезку листа.

Также на листе применяется метод бинаризации Отсу, для 
выделения букв и цифр.

### knn.py

Требования только openCV

алгоритм The k-nearest neighbors (KNN) 
простой и эффективный способ проверить работоспособность датасета

Разбивает изображения на блоки, присваивает им названия,
формирует тестовую последовательность.
на данный момент, точность 97 %

### cnn.py
Требования Tensorflow, Keras

Требования: opencv-contrib-python, numpy, scipy, imutils

алгоритм k-Nearest-Neighbour (kNN)

простой и эффективный способ проверить работоспособность алгоритма и ошибок датасете

Разбивает изображения на блоки, присваивает им названия,
формирует тестовую последовательность.
Точность 97 %

### cnn.py

Требования: Tensorflow, Keras.


алгоритм CNN

Принцип предобработки изображений тот же
