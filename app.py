import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from streamlit_option_menu import option_menu
import cohere
import time
import os
from dotenv import load_dotenv
load_dotenv()


# ============ API Keys and Email Config ============


COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

co = cohere.Client(COHERE_API_KEY)
# ============ Streamlit Config ============
st.set_page_config(page_title="Emotion Insight App", layout="wide")

# ============ Email Function ============
def send_email(name, email, message):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"📩 New Contact from {name}"

    body = f"Name: {name}\nEmail: {email}\nMessage:\n\n{message}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, email_password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully!"
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Please check your email and app password."
    except smtplib.SMTPConnectError:
        return False, "Connection failed. Please check your network or SMTP server settings."
    except Exception as e:
        return False, f"Failed to send message. Error: {str(e)}"

# ============ Sidebar Navigation ============
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "Project", "Chatbot", "Get in Touch"],
        icons=["house", "robot", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

# ============ Home Page ============
if selected == "Home":
    st.markdown("<h1 style='text-align: center; color: #FFD700;'> Embark on Emotion Insight App </h1>", unsafe_allow_html=True)
    
    with st.container():
        
        st.markdown("##  About the Project")
        
        st.markdown("""
        <div style="font-size: 17px; line-height: 1.6;">
        The <b>Emotion Insight App</b> is a comprehensive mental wellness platform that integrates AI and Deep Learning to understand and support users' emotional health. Built with the goal of democratizing emotional intelligence, this tool serves as a bridge between mental well-being and cutting-edge technology. It provides a space where people can reflect, analyze, and seek clarity about their emotional states using the following modules:<br><br>

        <ul>
            <li><b>Real-Time Facial Emotion Recognition</b> - Leveraging pretrained Convolutional Neural Networks (CNNs), our app detects emotions through facial images. By applying transfer learning on models like MobileNetV2, the platform achieves high accuracy across diverse lighting, angles, and occlusion scenarios. Users can either upload photos or later access real-time webcam-based detection.</li>
            <li><b>Natural Language Chatbot</b> - Powered by Cohere AI, this empathetic assistant engages users in meaningful conversations related to mental health. The chatbot is trained to understand context, suggest supportive replies, and maintain a natural dialogue, mimicking the behavior of a wellness counselor while being completely anonymous.</li>
            <li><b>Sentiment Analysis</b> - Users can input any form of text—journals, posts, or messages—and the system will evaluate sentiment strength and categorize emotional tone into categories such as joy, sadness, fear, surprise, anger, and more. This enables self-reflection and trend analysis over time.</li>
        </ul>

        Beyond tools, Emotion Insight App stands as a movement. With mental health issues rising globally—especially post-pandemic—this app acknowledges the urgency of proactive care. By blending visual and textual analysis, it offers a 360-degree emotional insight, helping users become more aware, balanced, and resilient.<br><br>

        <b>Tech Stack Overview:</b>
        <ul>
            <li><b>Frontend</b>: Streamlit for rapid, interactive UI development.</li>
            <li><b>Backend</b>: Python with integration of smtplib for feedback and contact forms.</li>
            <li><b>AI Models</b>: CNN-based emotion classifier, Cohere's generative NLP model.</li>
            <li><b>Deployment Ready</b>: Modular and scalable structure for cloud deployment (e.g., Streamlit Cloud, Heroku, GCP).</li>
        </ul>

        <b>Target Users:</b> Students, therapists, HR professionals, educators, and anyone curious about emotional patterns. The platform supports journaling, emotional wellness tracking, and AI-based guidance—all while protecting user privacy.

        <br><br>
        <b>Key Outcomes:</b><br>
        - Encourage self-awareness and emotional literacy in youth and adults alike.<br>
        - Reduce stigma around mental health by introducing non-intrusive, tech-enabled aids.<br>
        - Provide institutions a tool for integrating well-being into digital systems.<br><br>

        In future versions, we plan to implement emotion trend visualization dashboards, recommendation engines for therapy, real-time emotion tracking via webcam, and deeper personalization based on emotion history. With user feedback and open collaboration, Emotion Insight App aims to be the go-to emotional health companion for everyday mental resilience.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🚀 Key Features")
        with st.expander(" Facial Emotion Detection"):
            st.write("Detect emotions from facial expressions using images or webcam (coming soon).")
        with st.expander(" Mental Health Chatbot"):
            st.write("A smart, empathetic Cohere-powered chatbot that responds based on user input.")
        with st.expander(" Sentiment Analysis"):
            st.write("Understand underlying emotions in text using NLP and emotional lexicons.")
        with st.expander(" Contact Form"):
            st.write("A simple form to share your feedback, queries, or reach out for support.")

    with col2:
        st.markdown("### 🎯 Use Cases")
        st.success("🧘 Mental Health Monitoring:\n\nEarly identification of emotional distress to support mental health.")
        st.info("📊 Emotion Tracking:\n\nVisualize emotional trends over time (to be implemented).")
        st.warning("🧑‍🏫 Educational Tool:\n\nDemonstrate emotional intelligence using interactive tech.")
        st.error("💼 HR & Workplace Wellness:\n\nEncourage emotional well-being in organizational settings.")

    st.markdown("---")
    st.markdown("<div style='text-align:center; color: gray;'>Empowering minds, one emotion at a time.</div>", unsafe_allow_html=True)
# ============ Project Page ============
elif selected == "Project":
    import cv2
    import numpy as np
    from keras.models import load_model
    import os
    from PIL import Image

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
    st.write("This module combines facial emotion recognition with a mental health questionnaire.")

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
                from chatgpt_integration import analyze_responses  # make sure this file exists and is imported correctly
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

# ============ Get in Touch Page ============
elif selected == "Get in Touch":
    st.title("📬 Get in Touch with Us")
    st.markdown("### We'd love to hear from you! Please fill out the form below:")

    with st.form("contact_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Your Name", placeholder="Enter your full name")
        with col2:
            email = st.text_input("📧 Your Email", placeholder="Enter your email address")

        message = st.text_area("💬 Your Message", height=150, placeholder="Type your message here...")
        submitted = st.form_submit_button("🚀 Send Message")

        if submitted:
            if name and email and message:
                success, message_status = send_email(name, email, message)
                if success:
                    st.success(f"✅ {message_status}")
                else:
                    st.warning(f"⚠️ {message_status}")
            else:
                st.error("❗ Please fill in all fields.")

# ============ Chatbot Page ============
elif selected == "Chatbot":
    st.title("🤖 Chatbot – Powered by Cohere AI")
    st.write("Ask anything related to emotional well-being or mental health!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("🧑 You:", key="user_input")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = co.chat(
                    message=user_input,
                    chat_history=[
                        {"user_name": speaker, "text": text}
                        for speaker, text in st.session_state.chat_history
                    ]
                )
                bot_reply = response.text

                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Cohere", bot_reply))

            except Exception as e:
                st.error(f"💥 Error: {e}")

    for speaker, text in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑 {speaker}:** {text}")
        else:
            st.markdown(f"**🤖 {speaker}:** {text}")