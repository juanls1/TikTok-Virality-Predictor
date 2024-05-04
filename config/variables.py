# Path to the CSV file
csv_file = 'data/inputs/virality_info.csv'

# Path to the Audio CSV file
audiocsv_file = 'src/audio'


# Audio

## Speech extraction

ffmpeg_path = 'config/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe'  # Ruta completa al ejecutable de FFmpeg (Necesario tenerlo instalado para speech extraction)
wav_path = 'data/inputs/wav/'

## Data Exploration & Visualization

audio_path = 'data/inputs/audios/'
indiv_sample_path = 'data/inputs/audios/6875370613523909890_audio.mp3'


# Text

# Path to the text files
text_path = 'data/inputs/texts/'

# Path to the JSON file
json_file = 'data/inputs/tiktok_info.json'

# Path to the Text CSV file
textcsv_file = 'src/text'

# Language map from language codes to full language names
language_map = {
    'ar': 'arabic',
    'az': 'azerbaijani',
    'eu': 'basque',
    'bn': 'bengali',
    'ca': 'catalan',
    'zh-cn': 'chinese',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'fi': 'finnish',
    'fr': 'french',
    'de': 'german',
    'el': 'greek',
    'he': 'hebrew',
    'hi': 'hindi',
    'hu': 'hungarian',
    'id': 'indonesian',
    'it': 'italian',
    'kk': 'kazakh',
    'ne': 'nepali',
    'no': 'norwegian',
    'pt': 'portuguese',
    'ro': 'romanian',
    'ru': 'russian',
    'sl': 'slovene',
    'es': 'spanish',
    'sv': 'swedish',
    'tg': 'tajik',
    'tr': 'turkish'
}

# Path to the best model trained

model_path = 'data/outputs/texts/'