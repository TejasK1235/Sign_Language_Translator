from flask import Flask, render_template, request, jsonify
import mediapipe as mp
import numpy as np
import pickle
import cv2
import base64

app = Flask(__name__)


model_dict = pickle.load(open('artifacts/model.pkl', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Hello',1: 'Yes',2: 'No',3: 'Instant Transmission',4: 'LOL'}

def process_frame(image_data):
    # Decode base64 image to OpenCV format
    encoded_data = image_data.split(',')[1]
    decoded_data = base64.b64decode(encoded_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    temp, x_, y_ = [], [], []
    height, width, _ = frame.shape
    # Opencv uses the base image format as BGR but for literally everything else at backend
    # we need it as RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                temp.append(x)
                temp.append(y)
                x_.append(x)
                y_.append(y)

        try:
            preds = model.predict([np.array(temp)])
            pred_label = labels_dict[int(preds[0])]
            prediction = pred_label
        except:
            prediction = ""

    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    prediction = process_frame(image_data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
