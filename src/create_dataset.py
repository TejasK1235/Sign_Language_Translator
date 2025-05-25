import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle

DATA_DIR = './data'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)


data = []
labels = []
for dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir)):
        temp = []
        img = cv2.imread(os.path.join(DATA_DIR,dir,img_path))

        # Opencv has it's default image format as BGR and not RGB
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    temp.append(x)
                    temp.append(y)

            data.append(temp)
            labels.append(dir)


with open('data.pickle','wb') as f:
    pickle.dump({'data':data,'labels':labels},f)

#         plt.figure()
#         plt.imshow(img_rgb)

    
# plt.show()