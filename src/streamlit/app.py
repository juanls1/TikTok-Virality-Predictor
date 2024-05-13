import sys
from pathlib import Path
import streamlit as st

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from src.utils.utils_streamlit import extract_audio, clean_hashtags, create_text_prediction
from src.audio.audio_utils import transcribe_audio
from src.text.text_utils import clean_text


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
    
    # Campo de entrada de texto para el caption
    caption = st.text_input("Enter the caption (text below the video) (e.g., 'Check out my new dance moves!'):")

    # Campo de entrada de texto para los hashtags
    hashtags = st.text_input("Enter the hashtags (comma separated) (e.g., '#dance, #tiktok, #viral'):")

    if uploaded_file:
        # Procesamiento del video
        video_bytes = uploaded_file.read()
        
        # Aquí puedes llamar a tus funciones para extraer el audio, transcribirlo, y hacer la predicción
        audio_data, audio_temp = extract_audio(video_bytes)
        transcription = transcribe_audio(audio_temp)
        
        cleaned_transcription = clean_text(transcription)
        cleaned_caption = clean_text(caption)
        cleaned_hashtags = clean_hashtags(hashtags)
        cleaned_hashtags = ', '.join(cleaned_hashtags)
        
        if cleaned_caption.strip() == "":
            cleaned_caption = "None"
            
        if cleaned_hashtags.strip() == "":
            cleaned_hashtags = "None"
        
        if cleaned_transcription == "not understood" or cleaned_transcription == "error":
            text = f"Transcription: None. Caption: {cleaned_caption}. Hashtags: {cleaned_hashtags}. "
            st.write(f"Transcription {cleaned_transcription}")
        else:
            text = f"Transcription: {cleaned_transcription}. Caption: {cleaned_caption}. Hashtags: {cleaned_hashtags}. "
        
        
        # Configuraciones de la barra lateral
        with st.sidebar:
            st.header("Settings")
            # Selector de modo de clasificación
            regression_mode = st.radio(
                "Regression Mode:",
                ("Independent", "Mixed"),
                help="Select the independent mode to obtain a set of predictions for each model (text, image and audio), \
                or the mixed mode to obtain a single prediction using a multi-modal model."
            )
            
        # Prediction based on the selected mode
        if regression_mode == "Independent":
            
            text_prediction = create_text_prediction(text)
            # audio_model = load_model(model_paths["audio_model"])
            # image_model = load_model(model_paths["image_model"])
            
            
            # prediction_audio = audio_model.predict(audio_data)
            # prediction_image = image_model.predict(video_bytes)
            
            # Mostrar la predicción
            st.success(f"##### Text Virality Prediction: {text_prediction}")
            
            prediction = {"Text Virality Prediction": text_prediction, "image": "HEy", "audio": "HEy"}
            
        elif regression_mode == "Mixed":
            
            # multi_model = load_model(model_paths["multi_model"])
            
            # prediction_multi = multi_model.predict([transcription, audio_data, video_bytes])
            
            prediction_multi = {"text": text}
            
            prediction = {"multi": prediction_multi}
            
        # Mostrar la predicción
        st.write(prediction)
        

if __name__ == "__main__":
    main()