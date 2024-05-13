import sys
from pathlib import Path
import os
import csv 
import numpy as np

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path.cwd().resolve().parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

import subprocess
from langdetect import detect
import speech_recognition as sr
import sklearn.preprocessing
import struct
import librosa
from torch.utils.data import Dataset

from config.variables import ffmpeg_path

def convert_to_wav(mp3_file, output_folder):
    wav_file = os.path.splitext(os.path.basename(mp3_file))[0] + '.wav'
    wav_file_path = os.path.join(output_folder, wav_file)
    subprocess.run([os.path.join(root_dir, ffmpeg_path), '-i', mp3_file, '-ac', '1', '-ar', '16000', wav_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_file_path

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)  # Grabamos el audio del archivo
        try:
            language = detect_language(audio_file)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            print("No se pudo entender el audio")
            return "Not understood"
        except sr.RequestError as e:
            print(f"Error en la solicitud a Google Speech Recognition API: {e}")
            return "Error"
        

def detect_language(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            language = detect(recognizer.recognize_google(audio_data, show_all=True))
            return language
        except:
            return "en-EN"  # Establece un idioma predeterminado en caso de error

class WavFileHelper():
    
    def read_file_properties(self, filename):

        wave_file = open(filename,"rb")
        
        riff = wave_file.read(12)
        fmt = wave_file.read(36)
        
        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I",sample_rate_string)[0]
        
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H",bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)
    
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def open_csv_file(file_path, header):
    """
    Open a CSV file for writing and write the header.
    """
    file = open(file_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(header)
    return file, writer

def load_audio_features(audio_file):
    """
    Load audio file and extract features.
    """
    y, sr = librosa.load(audio_file, mono=True)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    features = [np.mean(feature) for feature in [chroma_stft, spec_cent, spec_bw, rolloff, zcr]]
    features += [np.mean(m) for m in mfcc]

    return features

def fetch_virality(video_id, df):
    """
    Fetch virality information from DataFrame based on video ID.
    """
    virality_row = df[df['id'] == video_id]
    if not virality_row.empty:
        return virality_row['norm_virality'].values[0]
    else:
        return 'N/A'
    
class AudioDataset(Dataset):
    def __init__(self, audio_folder, labels_df, feature_extractor):
        self.audio_folder = audio_folder
        self.labels_df = labels_df
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        audio_name = os.path.join(self.audio_folder, f"{self.labels_df.iloc[idx]['id']}_audio.wav")
        label = self.labels_df.iloc[idx]['norm_virality']
        
        y, sr = librosa.load(audio_name, sr = 16000)

        # Extract features using the model
        inputs = self.feature_extractor(y, sampling_rate=sr, return_tensors="pt")
        
        # Squeeze singleton dimension
        inputs["input_values"] = inputs["input_values"].squeeze(0)

        return inputs, label