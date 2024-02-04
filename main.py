import numpy as np
import streamlit as st
import cv2
import mediapipe as mp
from collections import Counter
import time

mp_face_mesh = mp.solutions.face_mesh

# Configuración de Streamlit
st.set_page_config(page_title="Deteccion de Emocion Facial", layout="wide")

# Configuración de MediaPipe
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)
emotions = ["Neutral", "Feliz", "Triste", "Sorprendido", "Enojado"]

def detectar_emociones(imagen):
    global emotions
    
    emotion_result = emotions[np.random.randint(0, len(emotions))]
    return emotion_result

def main():
    st.title("Deteccion de Emociones Facial")

    # Inicializar el estado de la sesión
    if 'counters' not in st.session_state:
        st.session_state.counters = Counter()

    # Configuración de la transmisión de video
    stframe = st.empty()

    # Definir la resolución deseada (ancho x alto)
    resolution = (300, 250)

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Configuración de las métricas de Streamlit
    metric_containers = {emotion: st.empty() for emotion in emotions}

    # Estado del botón
    start_button = st.button("Iniciar Cámara")
    stop_button = st.button("Detener Cámara")

    while True:
        if start_button:
            # Iniciar cámara
            ret, frame = video_capture.read()
            if not ret:
                st.error("No se puede acceder a la cámara")
                break

            # Convertir la imagen a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar el frame con MediaPipe
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # Obtener la emoción detectada
                emotion_result = detectar_emociones(frame)
                # Actualizar el contador de emociones
                st.session_state.counters[emotion_result] += 1
                # Dibujar el resultado en la imagen
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Emocion: {emotion_result}", (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Mostrar la imagen en Streamlit
            stframe.image(frame, channels="BGR", use_column_width=True)

        elif stop_button:
            # Detener cámara
            video_capture.release()
            # Reiniciar el contador cuando se detiene la cámara
            st.session_state.counters = Counter()

        # Mostrar los resultados en las métricas de Streamlit
        for emotion in emotions:
            metric_containers[emotion].metric(emotion, st.session_state.counters[emotion])

        # Pausa de 3 segundos
        time.sleep(3)

if __name__ == "__main__":
    main()
