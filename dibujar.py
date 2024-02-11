import cv2
import mediapipe as mp
import numpy as np
import random

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializa el objeto de captura de video
cap = cv2.VideoCapture(0)
# Aumentar la resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def draw_circle(image, center, radius, color, name):
    for i in range(15, 0, -1):
        cv2.circle(image, center, radius * i // 15, color, 2)
    cv2.putText(image, name, (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def draw_square(image, center, length, color, name):
    for i in range(15, 0, -1):
        top_left = (center[0] - length * i // 20, center[1] - length * i // 20)
        bottom_right = (center[0] + length * i // 20, center[1] + length * i // 20)
        cv2.rectangle(image, top_left, bottom_right, color, 2)
    cv2.putText(image, name, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando frames vacíos")
            continue

        # Convierte la imagen de BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Dibuja las anotaciones de la mano en la imagen.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calcula la distancia entre los dedos índices y pulgares
                index_distance = np.sqrt((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2)
                # Si la distancia es menor que un cierto umbral, dibuja un círculo o un cuadrado
                if index_distance < 0.25:
                    center = (int((index_finger_tip.x + thumb_tip.x) / 2 * image.shape[1]), int((index_finger_tip.y + thumb_tip.y) / 2 * image.shape[0]))
                    radius = int(index_distance * 1000)
                    print(radius, index_distance)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw_circle(image, center, radius, color, ' ')
                else:
                    center = (int((index_finger_tip.x + thumb_tip.x) / 2 * image.shape[1]), int((index_finger_tip.y + thumb_tip.y) / 2 * image.shape[0]))
                    length = int(index_distance * 1000)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw_square(image, center, length, color, ' ')

        cv2.imshow('jugando con MediaPipe', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
