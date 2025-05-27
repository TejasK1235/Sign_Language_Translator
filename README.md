# SignSpeak

**A real-time sign language translator**

SignSpeak is a machine learning–powered application that recognizes hand gestures in real-time using a webcam feed. Built with Python, MediaPipe, OpenCV, and Flask, the system provides a simple web interface where users can launch the translator and interact through sign language gestures.

---

## 🚀 Features

- Real-time gesture recognition using webcam  
- Interactive web interface built with Flask  
- Hand landmark detection powered by MediaPipe  
- Trained using a Random Forest Classifier (98% accuracy)  
- Lightweight and easy to run locally  

---

## 🗂️ Project Structure

```bash
SignSpeak/
│
├── artifacts/
│ ├── data.pickle # Refined training data (hand landmarks)
│ └── model.pkl # Trained RandomForestClassifier model
│
├── src/
│ ├── collect_images.py # Script to collect raw gesture images
│ ├── create_dataset.py # Extract hand landmarks using MediaPipe
│ └── train.py # Train the gesture recognition model
│
├── templates/
│ └── index.html # Home page with project info and Start button
│
├── .gitignore
├── README.md
├── app.py # Flask app to serve the interface and webcam feed
└── requirements.txt # List of dependencies
```


---

## ⚙️ How It Works

1. Visit the web interface served via Flask.  
2. Click the **Start** button on the homepage.  
3. A webcam window will pop up. Perform your gesture in front of the camera.  
4. Press `q` to close the webcam window and return to the homepage.

---

## 📊 Model Overview

- Training data: ~300–400 samples  
- Train/Test split: 80/20  
- Classifier used: `RandomForestClassifier` from Scikit-learn  
- Accuracy: **98%**  
- Other models tried:  
  - Decision Tree: ~70% accuracy  
  - Gradient Boosting: ~83% accuracy  

---

## 🛠 Requirements

Make sure you have the following Python libraries installed:

```bash
scikit-learn
mediapipe
flask
opencv-python
```

Install via pip:

```bash
pip install -r requirements.txt
```

## ▶️ Running the App
Simply run:

```bash
python app.py
Then open a browser and navigate to your local machine's IP Address
