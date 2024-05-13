import sys
from pathlib import Path
import os
import streamlit as st
import pandas as pd

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from src.utils.utils_streamlit import extract_audio, clean_hashtags, create_text_prediction, create_audio_prediction, extract_frames
from src.audio.audio_utils import transcribe_audio
from src.text.text_utils import clean_text


def main():
    # Configuración de la página
    st.set_page_config(page_title="ADNE - TikTok", layout="centered")
    st.title("TikTok virality prediction app 🚀")
    
    
    #Carga fichero csv de virality para seleccionar máximo y mínimo para desnormalizar
    df = pd.read_csv(os.path.join(root_dir, 'data/inputs/virality_info.csv'))
    
    min_val = df['virality'].min()
    max_val = df['virality'].max()

    
    # Mensaje de bienvenida
    with st.container():
        st.markdown("""
            Welcome to the TikTok virality prediction app, powered by Juan López and Ignacio Urretavizcaya.
            
            This app is powered by different Deep Learning models to predict the virality of your TikTok videos, using
            the content of the video as input. There are 2 modes available: independent and mixed prediction. The models
            used are based in LSTM, CNN and Transformer architectures.
        """)

    # Método para cargar el video
    uploaded_file = st.file_uploader("### Upload a video from TikTok", type=["mp4"])
    
    # Campo de entrada de texto para el caption
    caption = st.text_input("### Enter the caption (text below the video) (e.g., 'Check out my new dance moves!'):")

    # Campo de entrada de texto para los hashtags
    hashtags = st.text_input("### Enter the hashtags (comma separated) (e.g., '#dance, #tiktok, #viral'):")

    if uploaded_file:
        # Procesamiento del video
        video_bytes = uploaded_file.read()
        
        # Aquí puedes llamar a tus funciones para extraer el audio, transcribirlo, y hacer la predicción
        audio_temp = extract_audio(video_bytes)
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
        
        images = extract_frames(video_bytes)
        
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
            
            print(max_val, min_val)
            
            text_prediction = create_text_prediction(text)
            if text_prediction <= 0:
                text_prediction = 0.000000001
            desnormalized_text_prediction = round((text_prediction * (max_val - min_val) + min_val),0)
            text_prediction = round(text_prediction, 4)
            audio_prediction = create_audio_prediction(audio_temp)
            if audio_prediction <= 0:
                audio_prediction = 0.000000001
            desnormalized_audio_prediction = round((audio_prediction * (max_val - min_val) + min_val),0)
            audio_prediction = round(audio_prediction, 4)

            
            # Mostrar la predicción
            st.success(f"##### Text Virality Prediction: {desnormalized_text_prediction}. (Normalized: {text_prediction})")
            st.success(f"##### Audio Virality Prediction: {desnormalized_audio_prediction}. (Normalized: {audio_prediction})")
            

            
        elif regression_mode == "Mixed":
            
            # multi_model = load_model(model_paths["multi_model"])
            
            # prediction_multi = multi_model.predict([transcription, audio_data, video_bytes])
            
            prediction_multi = {"text": text}
            
        

if __name__ == "__main__":
    main()