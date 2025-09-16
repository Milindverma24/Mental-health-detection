import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model_path = "emotion_recognition_model.h5"
model = load_model(model_path)

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocess the image
def extract_features(image):
    feature = np.array(image, dtype="float32")
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels - add a fallback class for unknown predictions
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise',
    7: 'unknown'  # fallback for unexpected classes
}

# Start video capture
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set video dimensions
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)

        pred = model.predict(img, verbose=0)
        prediction_class = int(pred.argmax())
        prediction_label = labels.get(prediction_class, "unknown")

        color = (0, 255, 0) if prediction_label == "happy" else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()