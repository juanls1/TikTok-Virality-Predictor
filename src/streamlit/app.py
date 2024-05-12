import sys
from pathlib import Path
import streamlit as st
from keras.models import load_model


# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from utils.utils_streamlit import extract_audio, transcribe_audio
from config.variables import model_paths


def main():
    # Configuración de la página
    st.set_page_config(page_title="ADNE - TikTok", layout="centered")
    st.title("TikTok virality prediction app 🚀")
    
    # Mensaje de bienvenida
    with st.container():
        st.markdown("""
            Welcome to the TikTok virality prediction app, powered by Juan López and Ignacio Urretavizcaya.
            
            This app is powered by different Deep Learning models to predict the virality of your TikTok videos, using
            the content of the video as input. There are 2 modes available: independent and mixed prediction. The models
            used are based in LSTM, CNN and Transformer architectures.
        """)

    # Método para cargar el video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file:
        # Procesamiento del video
        video_bytes = uploaded_file.read()
        
        # Aquí puedes llamar a tus funciones para extraer el audio, transcribirlo, y hacer la predicción
        audio_data = extract_audio(video_bytes)
        transcription = transcribe_audio(audio_data)
        
        # Configuraciones de la barra lateral
        with st.sidebar:
            st.header("Conf")
            # Selector de modo de clasificación
            regression_mode = st.radio(
                "Regression Mode:",
                ("Independent", "Mixed"),
                help="Select the independent mode to obtain a set of predictions for each model (text, image and audio), \
                or the mixed mode to obtain a single prediction using a multi-modal model."
            )
            
        # Prediction based on the selected mode
        if regression_mode == "Independent":
            
            text_model = load_model(model_paths["text_model"])
            audio_model = load_model(model_paths["audio_model"])
            image_model = load_model(model_paths["image_model"])
            
            
            prediction_text = text_model.predict(transcription)
            prediction_audio = audio_model.predict(audio_data)
            prediction_image = image_model.predict(video_bytes)
            
            prediction = {"text": prediction_text, "image": prediction_image, "audio": prediction_audio}
            
        elif regression_mode == "Mixed":
            
            multi_model = load_model(model_paths["multi_model"])
            
            prediction_multi = multi_model.predict([transcription, audio_data, video_bytes])
            prediction = {"multi": prediction_multi}
            
        # Mostrar la predicción
        st.write(prediction)
        

if __name__ == "__main__":
    main()