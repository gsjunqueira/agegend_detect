""" Projeto de Detecção de Idade e Gênero utilizando OpenCV e Modelos Caffe.

Este projeto implementa um sistema de detecção de rostos, seguido da previsão de idade e gênero 
com base em redes neurais convolucionais pré-treinadas, usando o framework OpenCV e modelos Caffe.

O processo inclui:
    1. Detecção de rostos em tempo real usando um modelo de detecção de rostos baseado em SSD.
    2. Classificação do gênero (Masculino ou Feminino) para cada rosto detectado.
    3. Estimativa da faixa etária aproximada de cada rosto detectado.
    4. Exibição dos resultados em tempo real com as probabilidades calculadas.
"""
import os
import cv2
import numpy as np

DIR = 'agegend_detect/set/'

gender_proto = os.path.join(DIR, 'deploy_gender.prototxt')
gender_model = os.path.join(DIR, 'gender_net.caffemodel')
gender_list = ['Masculino', 'Feminino']
face_model = os.path.join(DIR, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
face_proto = os.path.join(DIR, 'deploy.prototxt.txt')
age_proto = os.path.join(DIR, 'deploy_age.prototxt')
age_model = os.path.join(DIR, 'age_net.caffemodel')
age_intervals = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)',
                 '(60, 100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FACE_NET = cv2.dnn.readNetFromCaffe(face_proto, face_model)
AGE_NET = cv2.dnn.readNetFromCaffe(age_proto, age_model)
GENDER_NET = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)


def get_faces(frame, conf_threshold=0.5):
    """ Detect faces in a frame using a pre-trained face detection model.

    Args:
        frame (numpy.ndarray): The input image frame where faces will be detected.
        conf_threshold (float, optional): Confidence threshold for detecting faces. Defaults to 0.5.

    Returns:
        list: A list of tuples containing the coordinates (start_x, start_y, end_x, end_y) 
        of each detected face.
    """
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    FACE_NET.setInput(blob)
    output = np.squeeze(FACE_NET.forward())
    faces = []

    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > conf_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1],
                                             frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            start_x = max(start_x, 0)
            start_y = max(start_y, 0)
            end_x = max(end_x, 0)
            end_y = max(end_y, 0)
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """ Resize an image to a specific width or height while maintaining the aspect ratio.

    Args:
        image (numpy.ndarray): The image to be resized.
        width (int, optional): The desired width. Defaults to None.
        height (int, optional): The desired height. Defaults to None.
        inter (int, optional): Interpolation method. Defaults to cv2.INTER_AREA.

    Returns:
        numpy.ndarray: The resized image.
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation = inter)


def gender_predict(face_img):
    """ Predict the gender of a detected face using a pre-trained gender classification model.

    Args:
        face_img (numpy.ndarray): Cropped face image.

    Returns:
        numpy.ndarray: Gender prediction probabilities.
    """
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    GENDER_NET.setInput(blob)
    return GENDER_NET.forward()


def age_predict(face_img):
    """ Predict the age range of a detected face using a pre-trained age classification model.

    Args:
        face_img (numpy.ndarray): Cropped face image.

    Returns:
        numpy.ndarray: Age range prediction probabilities.
    """
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    AGE_NET.setInput(blob)
    return AGE_NET.forward()


def predict_agegende():
    """ Capture video stream from the webcam, detect faces, and predict the age and gender 
    of each detected face. The results are displayed in real-time.

    This function uses a pre-trained face detection model to locate faces in each frame.
    It then uses pre-trained age and gender models to predict the age range and gender 
    of the faces detected.

    Press 'q' to quit the video stream.
    """
    cap = cv2.VideoCapture(2)

    while True:
        _, img = cap.read()
        frame = img.copy()
        if frame.shape[1] > FRAME_WIDTH:
            frame = image_resize(frame, width=FRAME_WIDTH)
        faces = get_faces(frame)
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            age_preds = age_predict(face_img)
            gender_preds = gender_predict(face_img)
            i = gender_preds[0].argmax()
            gender = gender_list[i]
            gender_conf_score = gender_preds[0][i]
            i = age_preds[0].argmax()
            age = age_intervals[i]
            age_conf_score = age_preds[0][i]
            label = f"{gender}-{gender_conf_score*100:.1f}%, {age}-{age_conf_score*100:.1f}%"
            print(label)
            y_pos = start_y - 15
            while y_pos < 15:
                y_pos += 15
            box_color = (255, 200, 0) if gender == "Masculino" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            cv2.putText(frame, label, (start_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, box_color, 2)

        cv2.imshow("Gender Estimator", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        # cv2.imwrite("output.jpg", frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_agegende()
