# ADNE Final Project: TikTok Sentiment Analysis

**Deep learning in Tik Tok by Juan López Segura and Ignacio Urretavizcaya Tato**

[Dataset Link](https://www.kaggle.com/datasets/erikvdven/tiktok-trending-december-2020/data)

This project focuses on applying deep learning techniques to TikTok videos to predict their virality, using video, text, and audio data.


## Previous requirements 📋

 1. Clone the repository

```
git clone https://github.com/juanls1/TikTok-sentiment-analysis.git
```

 2. Create a venv & install the necessary libraries 

```
python -m venv tiktok
tiktok\Scripts\activate
pip install -r requirements.txt

```

3. If necessary, update ```.gitignore``` with your own sensitive files


> [!IMPORTANT]
> If you want to train the models with the **GPU**, you'll have to install [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and the respectiove versions of [Torch and torchivision](https://pytorch.org/) manually, as the versions from the requirements.txt are for **CPU** training

## Folder Structure 📂

- `requirements.txt`: Libraries needed for the project

- `README.md`: This file

- `.gitignore`: Files to ignore in the repository

- **config**: Config files
    - **ffmpeg-master-latest-win64-gpl**: Neccesary files for audio treatment
    - `variables.py`: Variables for the project, divided by sections

- **data**: Contains the dataset used in the project
  - **inputs**:
    - **Manual Scrapping**: Folder containing manually extracted data (large in size)
    - **Raw**: Folder containing kaggle downloaded data (large in size)
    - **videos**: Videos used
    - **audios**: Audios used, extracted by us
    - **wav**: Audios used, extracted by us in wav format
    - **texts**: Text transcriptions of each video, extracted by us
    - `tiktok_info.json` with information about metrics, audio, creator, etc.
    -  `virality_info.csv` with calculated response metric information
  - **outputs**: Contains trained models (many not included due to size: https://drive.google.com/drive/folders/1w_4niCuYf2OcJYcZnwcHNgkU9kaeoEBb?usp=drive_link)
    - **audio**: Audio models
    - **texts**: Text models
    - **image**: Image models

- **mltools-0.1.42**: Custom library for the project, used normally in ICAI

- **src**: Contains the main code of the project

  - **text**: Text models
    - `text_study.ipynb`: Complete text analysis, preprocessing, models and results
    - `text_utils.py`: Text utility functions
    - `embeddings.csv`: Embeddings used for most text models

  - **audio**: Audio models
    - `audio_study.ipynb`: Complete audio analysis, preprocessing, models and results
    - `audio_utils.py`: Audio utility functions
    - `audio_features.csv`: Features used for most audio models

  - **image**: Image models
    - `image_study.ipynb`: Complete image analysis, preprocessing, models and results
    - `image_utils.py`: Image utility functions

  - **utils**: Main utility functions, separate from specific utilities: 
    - `utils_audio_extractor.py`: Audio extraction functions
    - `utils_streamlit.py`: Streamlit utility functions
    - `utils_TikTok_scrapper.py`: TikTok manual scrapper functions
    - `utils_json_formatting.py`: JSON formatting functions for the Kaggle RAW JSON
    - `utils_ids_counter.py`: IDs counter for the Kaggle RAW JSON

  - **streamlit**: Streamlit app
    - `app.py`: Streamlit app for the project

  - `project.ipynb`: Main file explaining the project, its steps and results.



## What can be done with the repository  🚀



## Future Improvement Work 🔧
