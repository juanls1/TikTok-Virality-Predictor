import sys
from pathlib import Path
import os
import streamlit as st
import pandas as pd

# Get the absolute path of the folder containing the module
root_dir = Path(__file__).resolve().parent.parent.parent

# Add the folder path to sys.path
sys.path.append(str(root_dir))

from src.utils.utils_streamlit import extract_audio, clean_hashtags, create_text_prediction, create_audio_prediction, extract_frames, create_image_prediction, CLIPRegressor, extract_multimodal_features, create_multimodal_prediction
from src.audio.audio_utils import transcribe_audio
from src.text.text_utils import clean_text


def main():
    # Page configuration
    st.set_page_config(page_title="ADNE - TikTok", layout="centered")
    st.title("TikTok virality prediction app 🚀")
    
    
    # Load virality csv file to select max and min for denormalization
    df = pd.read_csv(os.path.join(root_dir, 'data/inputs/virality_info.csv'))
    
    min_val = df['virality'].min()
    max_val = df['virality'].max()

    
    # Welcome message
    with st.container():
        st.markdown("""
            Welcome to the TikTok virality prediction app, powered by Juan López and Ignacio Urretavizcaya.
            
            This app is powered by different Deep Learning models to predict the virality of your TikTok videos, using
            the content of the video as input. There are 2 modes available: independent and mixed prediction. The models
            used are based in LSTM, CNN and Transformer architectures.
        """)

    # Method to upload the video
    uploaded_file = st.file_uploader("### Upload a video from TikTok", type=["mp4"])
    
    # Text input field for the caption
    caption = st.text_input("### Enter the caption (text below the video) (e.g., 'Check out my new dance moves!'):")

    # Text input field for the hashtags
    hashtags = st.text_input("### Enter the hashtags (comma separated) (e.g., '#dance, #tiktok, #viral'):")

    if uploaded_file:
        # Video processing
        video_bytes = uploaded_file.read()
        
        # Here you can call your functions to extract the audio, transcribe it, and make the prediction
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
        
        # Sidebar settings
        with st.sidebar:
            st.header("Settings")
            # Classification mode selector
            regression_mode = st.radio(
                "Regression Mode:",
                ("Independent", "Mixed"),
                help="Select the independent mode to obtain a set of predictions for each model (text, image and audio), \
                or the mixed mode to obtain a single prediction using a multi-modal model."
            )
            
        # Prediction based on the selected mode
        if regression_mode == "Independent":
            
            text_prediction = create_text_prediction(text)
            if text_prediction <= 0:
                text_prediction = 0.000000001
            desnormalized_text_prediction = int((text_prediction * (max_val - min_val) + min_val))
            text_prediction = round(text_prediction, 4)
            audio_prediction = create_audio_prediction(audio_temp)
            if audio_prediction <= 0:
                audio_prediction = 0.000000001
            desnormalized_audio_prediction = int((audio_prediction * (max_val - min_val) + min_val))
            audio_prediction = round(audio_prediction, 4)
            image_prediction = create_image_prediction(images)
            if image_prediction <= 0:
                image_prediction = 0.000000001
            desnormalized_image_prediction = int((image_prediction * (max_val - min_val) + min_val))
            image_prediction = round(image_prediction, 4)
    
            
            # Display the prediction
            st.success(f"##### Text Virality Prediction: {desnormalized_text_prediction}. (Normalized: {text_prediction})")
            st.success(f"##### Audio Virality Prediction: {desnormalized_audio_prediction}. (Normalized: {audio_prediction})")
            st.success(f"##### Image Virality Prediction: {desnormalized_image_prediction}. (Normalized: {image_prediction})")
            
            
        elif regression_mode == "Mixed":
            
            multimodal_text, multimodal_image = extract_multimodal_features(video_bytes, transcription, hashtags, caption)
            
            multimodal_prediction = create_multimodal_prediction(multimodal_text, multimodal_image)
            if multimodal_prediction <= 0:
                multimodal_prediction = 0.000000001
            desnormalized_multimodal_prediction = int((multimodal_prediction * (max_val - min_val) + min_val))
            multimodal_prediction = round(multimodal_prediction, 4)
            
            st.success(f"##### Multimodal Virality Prediction: {desnormalized_multimodal_prediction}. (Normalized: {multimodal_prediction})")
            

if __name__ == "__main__":
    main()
