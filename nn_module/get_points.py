import mediapipe as mp
import numpy as np
import cv2


def get_points(path, grap=True):
    mp_hands = mp.solutions.hands
    mp_Draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(path)
    result_points = np.zeros((21, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), dtype=object)  # массив куда запишем все точки
    count_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Обнаружение ключевых точек на руке

        with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, landmark in enumerate(hand_landmarks.landmark):

                    result_points[id][count_frame] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                    }
       
        if grap == True:
            mp_Draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Hand Landmarks", frame)
        count_frame += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return result_points, frame.shape[0], frame.shape[1]

a = get_points('C:\\python_folders\\mediapipe\\video3.MOV', grap = True )
