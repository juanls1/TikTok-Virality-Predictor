# ADNE Final Project: TikTok Sentiment Analysis

**Deep learning in Tik Tok by Juan López Segura and Ignacio Urretavizcaya Tato**

[Dataset Link](https://www.kaggle.com/datasets/erikvdven/tiktok-trending-december-2020/data)

This project focuses on applying deep learning techniques to TikTok videos to predict their virality, using video, text, and audio data.

> [!NOTE]
> 💥 **Important:** In this project, we define virality as a weighted combination of views, likes, comments, and shares, adjusted by their correlations. The formula to calculate virality is:
> 
> $$
> \text{Virality} = \text{views} + (1 - \text{corr}_{\text{views, likes}}) \times \text{likes} + (1 - \text{corr}_{\text{views, comments}}) \times \text{comments} + (1 - \text{corr}_{\text{views, shares}}) \times \text{shares}
> $$


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

This repository offers a powerful tool for analyzing the components that contribute to the virality of TikTok videos. With this repository, you can:

- Test New TikTok Videos: Use our Streamlit web page to upload a TikTok video and receive an immediate prediction on its potential for virality. This feature allows content creators to tweak their content based on the attributes of videos that have previously gone viral.

- Explore Viral Trends: Dive into the analytics of what makes a TikTok video successful. Our repository provides tools to dissect and analyze the key features of viral videos, such as visual appeal, audio engagement, and textual content, giving you insights into current trends in viral content.

- Expand our study: With a fork of this project you can try to explore the features that we have taken a lok into and try to extract a bit more performance than we did!

Whether you are a social media marketer, a content creator, or a data enthusiast, this repository gives you the tools to understand and predict the dynamics of viral TikTok videos. Start exploring today and enhance your content strategy with data-driven insights!


## Future Improvement Work 🔧

As our project continues to evolve, there are several areas where future work can greatly enhance the robustness and accuracy of our virality prediction model. Here are the key areas for improvement:

- Re-evaluating Virality Metrics: Our dataset includes a range of viral videos, with some outliers that are significantly more viral than others. These outliers can skew the predictive accuracy of our model. Future work could involve refining the virality metric to either adjust for these outliers or redefine what virality means in the context of our analysis. This will help achieve more balanced and representative predictions.

- Enhancing Image Analysis Models: The current image analysis component of our model has shown limitations, as evidenced by a consistently low validation loss of 0.01 across different models. This suggests that the model might not be capturing all relevant visual features effectively. Improving this aspect could involve exploring more sophisticated image processing algorithms or neural network architectures that are better suited for capturing the nuances of visual content that contribute to virality.

- Weight Adjustments in Model Integration: In our Streamlit app, the image analysis model currently holds less weight in the overall prediction due to its lower performance. Future improvements could focus on achieving a more balanced contribution from each model (image, audio, text) by enhancing the weaker models. This would provide a more holistic and accurate assessment of virality factors.

By addressing these areas, we aim to refine our approach and provide a more precise tool for predicting the virality of TikTok videos. This continuous improvement will not only enhance the utility of our repository but also contribute to the broader understanding of social media dynamics.


