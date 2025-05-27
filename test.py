import cv2
import mediapipe as mp
import numpy as np
import pickle

# Just a script to test the system without the flask app

model_dict = pickle.load(open('artifacts/model.pkl', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Hello',1: 'Yes',2: 'No',3: 'Instant Transmission',4: 'LOL'}

cap = cv2.VideoCapture(0)


while True:
    temp, x_, y_ = [], [], []

    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                temp.append(x)
                temp.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * width) - 10
        y1 = int(min(y_) * height) - 10
        x2 = int(max(x_) * width) - 10
        y2 = int(max(y_) * height) - 10

        try:
            preds = model.predict([np.array(temp)])
            pred_label = labels_dict[int(preds[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, pred_label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        except Exception:
            pass

    cv2.imshow('Sign Language Translator',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

