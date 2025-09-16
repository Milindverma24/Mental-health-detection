import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from chatgpt_integration import analyze_responses
import os
from PIL import Image
import tempfile

# Load the model
model_path = "emotion_recognition_model.h5"
if not os.path.exists(model_path):
    st.error(f"⚠️ Model file '{model_path}' not found.")
    st.stop()

model = load_model(model_path)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Labels for prediction
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocessing
def extract_features(image):
    feature = np.array(image, dtype="float32").reshape(1, 48, 48, 1) / 255.0
    return feature

def detect_emotion_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return "No face detected"
    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        img = extract_features(face)
        pred = model.predict(img, verbose=0)
        return labels[np.argmax(pred)]
    return "No face detected"

# UI Start
st.title("📸 Mental Health Analysis (Face + Text)")

st.write("## Step 1: Upload or Capture Image")
option = st.radio("Choose input method", ["Upload Image", "Use Webcam"])

uploaded_img = None
captured_img = None
detected_emotion = None

if option == "Upload Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_np = np.array(image.convert("RGB"))
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        detected_emotion = detect_emotion_from_image(img_cv2)
        st.success(f"🧠 Detected Emotion: **{detected_emotion}**")

elif option == "Use Webcam":
    camera_img = st.camera_input("Take a photo")
    if camera_img:
        img_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_bytes, 1)
        st.image(img_cv2, caption="Captured Photo", channels="BGR", use_column_width=True)
        detected_emotion = detect_emotion_from_image(img_cv2)
        st.success(f"🧠 Detected Emotion: **{detected_emotion}**")

# Step 2 - Text Questions
st.write("## Step 2: Mental Health Questionnaire")

questions = [
    "How would you describe your current emotional state?",
    "What are the primary stressors affecting you recently?",
    "Have you experienced persistent anxiety or stress in the past two weeks?",
    "Have you had trouble sleeping or noticed changes in your sleep pattern?",
    "Are you still interested in activities you previously enjoyed?",
    "Have you felt socially disconnected or isolated?",
    "Can you share a recent experience that made you feel positive?",
    "Have you recently experienced hopelessness or distressing thoughts?"
]

responses = []
all_answered = True

for i, q in enumerate(questions):
    ans = st.text_area(q, key=f"q_{i}")
    if not ans.strip():
        all_answered = False
    responses.append(ans)

if st.button("🧠 Generate Final Mental Health Insight"):
    if not all_answered:
        st.warning("⚠️ Please answer all questions.")
    elif not detected_emotion or detected_emotion == "No face detected":
        st.warning("⚠️ Please provide a valid image with a detectable face.")
    else:
        st.write("🔍 Analyzing responses...")

        try:
            chat_summary = analyze_responses(responses)
            final_insight = (
                f"🧠 **Combined Mental Health Summary**\n\n"
                f"**Facial Emotion (30%)**: Detected emotion is **{detected_emotion}**.\n\n"
                f"**Chat-Based Summary (70%)**:\n\n{chat_summary}\n"
            )

            st.markdown(final_insight)

            if "consult" in chat_summary.lower() or "professional help" in chat_summary.lower():
                st.error("⚠️ Recommendation: Consider consulting a mental health professional.")
            else:
                st.success("✅ You're showing signs of stability, but always check in with yourself regularly.")

        except Exception as e:
            st.error("⚠️ Error analyzing responses. Check your API or integration.")
            st.code(str(e))