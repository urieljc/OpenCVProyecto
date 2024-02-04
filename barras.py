import streamlit as st
import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to process the frame and detect face mesh
def detect_face_mesh(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(frame_rgb)

    # Draw face mesh on the frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

    return frame

# Streamlit Application
def main():
    st.title("Face Mesh Application")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Main loop
    while st.checkbox("Capture Face Mesh", key="start_stop"):
        ret, frame = cap.read()

        # Detect and display face mesh
        frame_with_mesh = detect_face_mesh(frame)

        # Display the frame with face mesh using Streamlit
        st.image(frame_with_mesh, channels="BGR", use_column_width=True)

    # Release resources
    cap.release()

if __name__ == "__main__":
    main()
